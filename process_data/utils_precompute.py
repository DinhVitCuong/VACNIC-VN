"""Convert raw ViWiki JSON files into the format consumed by
``ViWiki_face_ner_match.py`` and additionally compute the image and text
features described in that reader's documentation.

The input directory must contain ``train.json``, ``val.json`` and ``test.json``
with entries of the form::

    {
        "0": {
            "image_path": "/path/to/img.jpg",
            "paragraphs": [...],
            "scores": [...],
            "caption": "caption text",
            "context": ["sentence 1", "sentence 2", ...]
        },
        ...
    }

For each item this script will:

* copy the image to ``--image-out`` (if provided)
* detect faces with **MTCNN** and extract **FaceNet** embeddings with max faces = 4
* detect objects using **YoloV8** and encode them with **ResNet152** We filter out objects with a confidence less than 0.3 and select up to 64 objects
* extract a global image feature with **ResNet152**
* embed the article text using a **RoBERTa** model (PhoBERT for Vietnamese)

The resulting ``splits.json``, ``articles.json`` and ``objects.json`` match the
schema in :mod:`tell.data.dataset_readers.ViWiki_face_ner_match`.  ``splits``
contains the face embeddings and global image features while object features are
stored in ``objects.json``.
"""

import os
from typing import Tuple, List
import torch
from PIL import Image
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from ultralytics import YOLO
from typing import Optional, Tuple, List
import py_vncorenlp

Image.MAX_IMAGE_PIXELS = None
SHARD_THRESHOLD = 3000
SHARD_SIZE = 2000

def _rel_or_abs(path: str, base_dir: str, use_abs: bool = False) -> str:
    return path if (use_abs or os.path.isabs(path)) else os.path.relpath(path, start=base_dir)

def setup_models(device: torch.device, vncorenlp_path):
    print("[DEBUG] SETTING UP MODELS")
    # --- VnCoreNLP ---
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx15g'
    )
    print("[DEBUG] LOADED VNCORENLP!")
    # --- Face detection + embedding ---
    mtcnn = MTCNN(keep_all=True, device=str(device))
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("[DEBUG] LOADED FaceNet!")

    # --- Global / object image features ---
    weights = ResNet152_Weights.IMAGENET1K_V1
    base = resnet152(weights=weights).eval().to(device)
    resnet = nn.Sequential(*list(base.children())[:-2]).eval().to(device)
    resnet_object = nn.Sequential(*list(base.children())[:-1]).eval().to(device)
    print("[DEBUG] LOADED ResNet152!")

    # --- YOLOv8 ---
    yolo = YOLO("yolov8m.pt")
    yolo.fuse()
    print("[DEBUG] LOADED YOLOv8!")

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    return {
        "vncore": vncore,
        "mtcnn": mtcnn,
        "facenet": facenet,
        "resnet": resnet,
        "resnet_object": resnet_object,
        "yolo": yolo,
        "preprocess": preprocess,
        "device": device,
    }


def _pad_to_len(x: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [S, D] on any device; returns (x_padded [target_len, D], mask [target_len] with True=PAD)
    If S >= target_len -> truncate and mask all False.
    """
    S, D = x.shape
    if S >= target_len:
        return x[:target_len], torch.zeros(target_len, dtype=torch.bool, device=x.device)
    pad = torch.zeros(target_len - S, D, dtype=x.dtype, device=x.device)
    out = torch.cat([x, pad], dim=0)
    mask = torch.zeros(target_len, dtype=torch.bool, device=x.device)
    mask[S:] = True
    return out, mask

def extract_faces_emb(
    image_path: str,
    mtcnn,
    facenet,
    device: torch.device,
    max_faces: int = 4,
    pad_to: Optional[int] = None,   # set to an int (e.g., 4) if you want fixed length
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: [S, 512] float32 (CPU)
      mask:  [S] bool (True=PAD)
    """
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        faces, probs = mtcnn(img, return_prob=True)

    if faces is None or len(faces) == 0:
        feats = torch.zeros((0, 512), dtype=torch.float32)
        if pad_to is not None:
            return _pad_to_len(feats, pad_to)
        return feats, torch.zeros((0,), dtype=torch.bool)

    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()

    facelist = sorted(zip(faces, probs), key=lambda x: x[1], reverse=True)[:max_faces]
    face_tensors = torch.stack([fp[0] for fp in facelist]).to(device)  # [k,3,160,160]
    with torch.no_grad():
        embeds = facenet(face_tensors).float()  # [k,512] on device
    feats = embeds.detach().cpu().contiguous()

    if pad_to is not None:
        return _pad_to_len(feats, pad_to)
    return feats, torch.zeros(feats.size(0), dtype=torch.bool)

def extract_objects_emb(
    image_path: str,
    yolo, resnet, preprocess, device: torch.device,
    conf: float = 0.3, iou: float = 0.45, max_det: int = 64,
    pad_to: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: [S, 2048] float32 (CPU)
      mask:  [S] bool (True=PAD)
    """
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        results = yolo(image_path, conf=conf, iou=iou, max_det=max_det, verbose=False, show=False)

    dets: List[torch.Tensor] = []
    if results:
        res = results[0]
        xyxy = res.boxes.xyxy.cpu()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
            t = preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(t).squeeze().detach().float()  # [2048]
            dets.append(feat.cpu())

    feats = torch.stack(dets, dim=0) if len(dets) else torch.zeros((0, 2048), dtype=torch.float32)
    if pad_to is not None:
        return _pad_to_len(feats, pad_to)
    return feats, torch.zeros(feats.size(0), dtype=torch.bool)
