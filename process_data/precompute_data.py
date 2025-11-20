import json
import os
import re
import numpy as np
from utils_precompute import  setup_models, extract_faces_emb, extract_entities
from tqdm import tqdm
import py_vncorenlp
import logging  # Added for better error handling

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print('Start process')


def is_abbreviation(entity):
    """
    Check if a string is an abbreviation like A.B.
    Pattern: A. B. (may include a name afterward)
    """
    pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
    return re.match(pattern, entity) is not None

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

def process_dataset(input_json_path, output_json_path, models):
    """
    Read data from input_json_path, process only new entries, and update output_json_path.
    """
    # Load existing output JSON if it exists
    new_data = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            logger.info(f"Loaded {len(new_data)} existing entries from {output_json_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error loading existing JSON {output_json_path}: {e}")
            new_data = {}

    # Load input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_count = 0
    mtcnn = models["mtcnn"]; facenet = models["facenet"];
    device = models["device"]; vncore = models["vncore"]

    for hash_id, content in tqdm(data.items(), desc="Processing dataset"):
        # Skip if already processed
        if hash_id in new_data:
            continue

        try:
            new_entry = {}
            img_id = content["image_path"].split("images/")[-1]
            image_path = fr"/datastore/npl/ICEK/Wikipedia/images_resized_refine/{img_id}"

            # Extract faces
            faces_embbed, _ = extract_faces_emb(image_path, mtcnn, facenet, device)
            face_emb_path = os.path.join(r"/datastore/npl/ICEK/vacnic/data/embedding/faces", f"{hash_id}.npy")
            np.save(face_emb_path, faces_embbed)
            new_entry["face_emb_dir"] = face_emb_path

            list_sents_byclip = content.get("paragraphs", [])
            sents_byclip = '. '.join(list_sents_byclip)

            # Extract entities from     context
            context_entities = extract_entities(sents_byclip, vncore)
            names_art = list(context_entities.get("PERSON", []))
            org_norp_art = list(context_entities.get("ORGANIZATION", [])) + list(context_entities.get("NORP", []))
            gpe_loc_art = list(context_entities.get("GPE", [])) + list(context_entities.get("LOCATION", []))

            # Extract entities from captions
            caption = content["caption"]
            captions_text = caption
            caption_entities = extract_entities(captions_text, vncore)
            names_caption = list(caption_entities.get("PERSON", []))
            org_norp_caption = list(caption_entities.get("ORGANIZATION", [])) + list(caption_entities.get("NORP", []))
            gpe_loc_caption = list(caption_entities.get("GPE", [])) + list(caption_entities.get("LOCATION", []))

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
            processed_count += 1

            # Optional: Periodic save every 100 entries for crash recovery
            if processed_count % 100 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved {processed_count} new entries at {hash_id}")

        except Exception as e:
            logger.error(f"Error processing {hash_id}: {e}")

    # Final save
    if processed_count > 0:  # Only write if new data was added
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data processed and saved to {output_json_path}, added {processed_count} new entries")
    else:
        logger.info(f"No new entries to process for {output_json_path}")

if __name__ == "__main__":

    datasets = [
        (r"/datastore/npl/ICEK/Wikipedia/content/ver5/demo20.json", r"/datastore/npl/ICEK/vacnic/data/demo20.json"),
        (r"/datastore/npl/ICEK/Wikipedia/content/ver5/val.json", r"/datastore/npl/ICEK/vacnic/data/val.json"),
        (r"/datastore/npl/ICEK/Wikipedia/content/ver5/test.json", r"/datastore/npl/ICEK/vacnic/data/test.json"),
        (r"/datastore/npl/ICEK/Wikipedia/content/ver5/train.json", r"/datastore/npl/ICEK/vacnic/data/train.json"),
    ]
    device="cuda"
    vncore_path = r"/datastore/npl/ICEK/VnCoreNLP"
    models = setup_models(device, vncore_path)
    for input_json, output_json in datasets:
        process_dataset(input_json, output_json, models)