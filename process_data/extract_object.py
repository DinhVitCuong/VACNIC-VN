import torch
from PIL import Image
import os
import requests
from io import BytesIO
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import clip
from torchvision import transforms

# Constants
MAX_PIXELS = 89478485
RESAMPLING = Image.LANCZOS

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load Faster R-CNN for object detection
obj_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
obj_detection_model.to(device)
obj_detection_model.eval()

def load_image(path_or_url):
    try:
        # Temporarily disable pixel limit
        original_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None

        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/18.19041"}
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(path_or_url).convert('RGB')

        # Check and resize if necessary
        num_pixels = image.width * image.height
        if num_pixels > MAX_PIXELS:
            print(f"Reducing image size from {image.width}x{image.height} pixels.")
            scaling_factor = (MAX_PIXELS / num_pixels) ** 0.5
            new_width = max(1, int(image.width * scaling_factor))
            new_height = max(1, int(image.height * scaling_factor))
            image = image.resize((new_width, new_height), RESAMPLING)
            print(f"Image size after reduction: {new_width}x{new_height} pixels.")

        # Restore pixel limit
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        return image
    except Exception as e:
        print(f"Error loading image from {path_or_url}: {e}")
        return None

def detect_objects(image_path, image_url):
    img = load_image(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}.")
        print(f"Attempting to load from {image_url}")
        img = load_image(image_url)
        if img is None:
            print("Failed to load image from URL as well.")
            return [], []
    img_t = transforms.ToTensor()(img).to(device)
    with torch.no_grad():
        predictions = obj_detection_model([img_t])
    score_threshold = 0.5
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    indices = scores > score_threshold
    boxes = boxes[indices].cpu().int().tolist()
    labels = labels[indices].cpu().tolist()
    return boxes, labels

def extract_object_embedding(image_path, image_url, bbox):
    img = load_image(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}.")
        print(f"Attempting to load from {image_url}")
        img = load_image(image_url)
        if img is None:
            print("Failed to load image from URL as well.")
            return []
    obj_img = img.crop(bbox)
    obj_img = clip_preprocess(obj_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(obj_img)
    embedding_np = embedding.cpu().numpy().squeeze()
    return embedding_np