import json
import os
from collections import defaultdict
import re
import numpy as np
from extract_face import extract_faces, get_face_embedding
from extract_object import detect_objects, extract_object_embedding
from tqdm import tqdm
import py_vncorenlp

print('Start process')

# Initialize vncorenlp
# vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir="/data2/npl/ICEK/VnCoreNLP")
py_vncorenlp.download_model(save_dir="/data2/npl/ICEK/VnCoreNLP")
vncore = py_vncorenlp.VnCoreNLP(
    annotators=["wseg", "pos", "ner", "parse"],
    save_dir="/data2/npl/ICEK/VnCoreNLP",
    max_heap_size='-Xmx10g'
)

def preprocess(text):
    cleaned_text = re.sub(r'\(Ảnh.*?\)', '', text)
    return cleaned_text

def is_abbreviation(entity):
    """
    Check if a string is an abbreviation like A.B.
    Pattern: A. B. (may include a name afterward)
    """
    pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
    return re.match(pattern, entity) is not None

def segment_text(text: str, model) -> str:
    """Segment text using VnCoreNLP and join sentences with a separator"""
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return ""
    segmented_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            segmented = model.word_segment(sent)[0]
            segmented_sentences.append(segmented)
        except Exception as e:
            logging.error(f"Error segmenting text: {e}")
            segmented_sentences.append(sent)
    return " <SEP> ".join(segmented_sentences)

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
        "GPE": "GPE", "B-GPE": "GPE", "I-GPE": "GPE",
        "NORP": "NORP", "B-NORP": "NORP", "I-NORP": "NORP",
        "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
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
                    entities[ent_type].add(' '.join(ent_text.split('_')).strip("•"))
    return {typ: sorted(vals) for typ, vals in entities.items()}

def find_name_positions(caption, names):
    """
    Find positions of names in caption.
    Return a list of [start, end] pairs.
    """
    positions = []
    for name in names:
        start = caption.find(name)
        while start != -1:
            end = start + len(name)
            positions.append([start, end])
            start = caption.find(name, end)
    return positions

def process_dataset(input_json_path, output_json_path, vncore):
    """
    Read data from input_json_path, process it, and save to output_json_path.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for hash_id, content in tqdm(data.items(), desc="Processing dataset"):
        new_entry = {}
        img_id = content["image_path"].split("images/")[-1]
        image_path = f"/data2/npl/ICEK/Wikipedia/images_resized/{img_id}"
        image_url = ""

        # Extract faces
        faces = extract_faces(image_path, image_url)
        faces_embbed = []
        if faces:
            for face in faces:
                face_emb = get_face_embedding(face)
                faces_embbed.append(face_emb)
            face_emb_path = os.path.join("/data2/npl/ICEK/vacnic/data/embeddings/faces", f"{hash_id}.npy")
            np.save(face_emb_path, faces_embbed)
            new_entry["face_emb_dir"] = face_emb_path
        else:
            new_entry["face_emb_dir"] = []

        # Extract objects
        objects, _ = detect_objects(image_path, image_url)
        objects_embbed = []
        if objects:
            for obj in objects:
                object_emb = extract_object_embedding(image_path, image_url, obj)
                objects_embbed.append(object_emb)
            object_emb_path = os.path.join("/data2/npl/ICEK/vacnic/data/embeddings/objects", f"{hash_id}.npy")
            np.save(object_emb_path, objects_embbed)
            new_entry["obj_emb_dir"] = object_emb_path
        else:
            new_entry["obj_emb_dir"] = []

        caption = content["caption"]
        list_sents_byclip = content.get("paragraphs", [])
        sents_byclip = ' '.join(list_sents_byclip)

        # Extract entities from context
        context_entities = extract_entities(sents_byclip, vncore)
        # print(f"[DEBUG] context_entities: {context_entities}")
        names_art = context_entities.get("PERSON", [])
        org_norp_art = context_entities.get("ORGANIZATION", []) + context_entities.get("NORP", [])
        gpe_loc_art = context_entities.get("GPE", []) + context_entities.get("LOCATION", [])

        # Extract entities from captions
        captions_text = caption
        caption_entities = extract_entities(captions_text, vncore)
        names_caption = caption_entities.get("PERSON", [])
        org_norp_caption = caption_entities.get("ORGANIZATION", []) + caption_entities.get("NORP", [])
        gpe_loc_caption = caption_entities.get("GPE", []) + caption_entities.get("LOCATION", [])

        # Add ner_cap field
        ner_cap = []
        for ent_type, entities in caption_entities.items():
            ner_cap.extend(entities)
        new_entry["ner_cap"] = list(set(ner_cap))

        # Create named_entities field
        named_entites = []
        for ent_type, entities in context_entities.items():
            named_entites.extend(entities)
        for ent_type, entities in caption_entities.items():
            named_entites.extend(entities)
        new_entry["named_entites"] = list(set(named_entites))

        # Add unique entity fields
        new_entry["names_art"] = list(set(names_art))
        new_entry["org_norp_art"] = list(set(org_norp_art))
        new_entry["gpe_loc_art"] = list(set(gpe_loc_art))
        new_entry["org_norp_cap"] = list(set(org_norp_caption))
        new_entry["gpe_loc_cap"] = list(set(gpe_loc_caption))

        # Combine context and caption entities
        new_entry["names"] = list(set(names_art + names_caption))
        new_entry["org_norp"] = list(set(org_norp_art + org_norp_caption))
        new_entry["gpe_loc"] = list(set(gpe_loc_art + gpe_loc_caption))

        # Add existing fields
        new_entry["image_path"] = image_path
        new_entry["caption"] = caption
        name_pos_cap = find_name_positions(caption, names_caption)
        if not name_pos_cap:
            name_pos_cap = []
        new_entry["name_pos_cap"] = name_pos_cap
        new_entry["sents_byclip"] = "\n\n".join(list_sents_byclip)

        new_data[hash_id] = new_entry

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Data processed and saved to {output_json_path}")

if __name__ == "__main__":
    input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/demo10.json"
    output_json = r"/data2/npl/ICEK/vacnic/data/demo10.json"
    process_dataset(input_json, output_json, vncore)
    # input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/val.json"
    # output_json = r"/data2/npl/ICEK/vacnic/data/val.json"
    # process_dataset(input_json, output_json, vncore)
    # input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/test.json"
    # output_json = r"/data2/npl/ICEK/vacnic/data/test.json"
    # process_dataset(input_json, output_json, vncore)
    # input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/train.json"
    # output_json = r"/data2/npl/ICEK/vacnic/data/train.json"
    # process_dataset(input_json, output_json, vncore)