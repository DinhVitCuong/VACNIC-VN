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
from collections import defaultdict
import re
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import Compose, ToTensor, Normalize
from ultralytics import YOLO
from typing import Optional, Tuple, List
from vncore_singleton import get_vncore

Image.MAX_IMAGE_PIXELS = None
SHARD_THRESHOLD = 3000
SHARD_SIZE = 2000

def setup_models(device: torch.device, vncorenlp_path):
    print("[DEBUG] SETTING UP MODELS")
    # --- VnCoreNLP ---
    vncore = get_vncore(vncorenlp_path, with_heap=True)
    print("[DEBUG] LOADED VNCORENLP!")
    # --- Face detection + embedding ---
    mtcnn = MTCNN(keep_all=True, device=str(device))
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("[DEBUG] LOADED FaceNet!")

    return {
        "vncore": vncore,
        "mtcnn": mtcnn,
        "facenet": facenet,
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

def _has_alnum(s: str) -> bool:
    # keep tokens that have at least one letter/number (works with Unicode)
    return any(ch.isalnum() for ch in s)

def extract_entities(text: str,
                     model
                    ):
    """
    Chia text thành từng câu, chạy NER trên mỗi câu bằng VnCoreNLP,
    rồi gộp kết quả (loại trùng). Nếu một câu lỗi hoặc không có entity,
    nó sẽ được bỏ qua.
    """
    # Define label mapping for vncorenlp NER labels
    label_mapping = {
        "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
        "ORG": "ORGANIZATION", "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
        "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION",
        # "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
    }
    entities = defaultdict(set)
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return {}
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            annotated_text = model.annotate_text(sent)
        except Exception as e:
            print(f"Lỗi khi annotating text: {e}")
            return entities

        for subsent in annotated_text:

            for word in annotated_text[subsent]:
                ent_type = label_mapping.get(word.get('nerLabel', ''), '')
                ent_text = word.get('wordForm', '').strip()
                if ent_type and ent_text:
                    raw_ent_text = (' '.join(ent_text.split('_')).strip("•"))
                    refined_ent_text = re.sub(r"[()\[\]{}'\"“”‘’]", "", raw_ent_text).strip()
                    if not ent_type or not ent_text or not _has_alnum(ent_text):
                        continue
                    entities[ent_type].add(refined_ent_text)
    return {typ: sorted(vals) for typ, vals in entities.items()}
