import json
import os
from collections import defaultdict
import re
import numpy as np
from extract_face import extract_faces, get_face_embedding
from extract_object import detect_objects, extract_object_embedding
from tqdm import tqdm


print('Start process')

def process_dataset(input_json_path, output_jsonl_path):
    """
    Đọc dữ liệu từ input_json_path, xử lý và lưu vào output_jsonl_path dưới định dạng JSONL.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for hash_id, content in tqdm(data.items(), desc="Đang xử lý dataset"):
            new_entry = {}

            image_path = content.get("image_path", "").replace(
                "/data/npl/ICEK/Wikipedia/images", "/data/npl/ICEK/DATASET/images")
            image_url = content.get("image_url", "")
            new_entry = content.copy()

            # Trích xuất face
            faces = extract_faces(image_path, image_url)
            faces_embbed = []
            if faces:
                for face in faces:
                    face_emb = get_face_embedding(face)
                    faces_embbed.append(face_emb)

                face_emb_path = os.path.join(
                    "/data/npl/ICEK/DATASET/content/vacnic/faces", f"{hash_id}.npy")
                np.save(face_emb_path, faces_embbed)
                new_entry["face_emb_dir"] = face_emb_path
            else:
                new_entry["face_emb_dir"] = []

            # Trích xuất objects
            objects, _ = detect_objects(image_path, image_url)
            objects_embbed = []
            if objects:
                for obj in objects:
                    object_emb = extract_object_embedding(image_path, image_url, obj)
                    objects_embbed.append(object_emb)

                object_emb_path = os.path.join(
                    "/data/npl/ICEK/DATASET/content/vacnic/objects", f"{hash_id}.npy")
                np.save(object_emb_path, objects_embbed)
                new_entry["obj_emb_dir"] = object_emb_path
            else:
                new_entry["obj_emb_dir"] = []

            new_entry["hash_id"] = hash_id  # Đưa hash_id vào mỗi dòng JSONL
            out_f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    print(f"Dữ liệu đã được xử lý và lưu vào {output_jsonl_path}")


if __name__ == "__main__":
    input_json = r"/data/npl/ICEK/DATASET/content/vacnic/test_vacnic.json"
    output_jsonl = r"/data/npl/ICEK/DATASET/content/vacnic/test_vacnic_final.json"

    process_dataset(input_json, output_jsonl)
