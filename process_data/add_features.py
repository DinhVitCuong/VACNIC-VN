# import json
# import os
# from collections import defaultdict
# import re
# import numpy as np
# from extract_face import extract_faces, get_face_embedding
# from extract_object import detect_objects, extract_object_embedding
# # from clip_get_sentences import retrieve_relevant_sentences, get_article
# from tqdm import tqdm
# import stanza
# from collections import defaultdict


# print('Start process')

# def preprocess(text):
#     cleaned_text = re.sub(r'\(Ảnh.*?\)', '', text)
#     return cleaned_text

# stanza.download('vi')
# nlp = stanza.Pipeline('vi', processors='tokenize,ner', use_gpu=True)

# def is_abbreviation(entity):
#     """
#     Kiểm tra xem một chuỗi có phải viết tắt theo dạng A.B. hay không.
#     Pattern: A. B. (có thể thêm tên tiếp theo sau)
#     """
#     pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
#     return re.match(pattern, entity) is not None

# def extract_entities(text, nlp):
#     """
#     Sử dụng Stanza để trích xuất các thực thể từ văn bản.
#     Đồng thời áp dụng kiểm tra chữ viết tắt để chỉ thêm vào kết quả 
#     nếu là abbreviation hợp lệ hoặc là thực thể bình thường không chứa '.'.
    
#     Trả về một dictionary với các loại thực thể (PERSON, ORGANIZATION, v.v.),
#     mỗi loại là một danh sách (list) các thực thể duy nhất.
#     """
#     entities = defaultdict(set)  
#     if not text.strip():
#         return {}

#     try:
#         doc = nlp(text)
#     except Exception as e:
#         print(f"Lỗi khi xử lý text: {e}")
#         return {}

#     # Ánh xạ nhãn từ Stanza sang nhãn chuẩn nếu cần
#     label_mapping = {
#         "PER": "PERSON",
#         "ORG": "ORGANIZATION",
#         "LOC": "LOCATION",
#         "GPE": "GPE",
#         "NORP": "NORP",
#         "MISC": "MISC"
#     }

#     # Duyệt qua từng câu và từng thực thể trong câu
#     for sentence in doc.sentences:
#         for ent in sentence.ents:
#             mapped_type = label_mapping.get(ent.type, ent.type)
#             ent_text = ent.text.strip()

#             if mapped_type and ent_text:
#                 # Kiểm tra xem thực thể có chứa dấu chấm (.)
#                 if '.' in ent_text:
#                     # Chỉ thêm nếu đúng là chữ viết tắt
#                     if is_abbreviation(ent_text):
#                         entities[mapped_type].add(ent_text)
#                 else:
#                     entities[mapped_type].add(ent_text)

#     # Chuyển set về list để xuất kết quả
#     return {key: list(value) for key, value in entities.items()}

# def find_name_positions(caption, names):
#     """
#     Tìm vị trí của các tên trong caption.
#     Trả về danh sách các cặp [start, end].
#     """
#     positions = []
#     for name in names:
#         start = caption.find(name)
#         while start != -1:
#             end = start + len(name)
#             positions.append([start, end])
#             start = caption.find(name, end)  
#     return positions
# # # code đã hoàn thiện, không chỉnh sửa
# def process_dataset(input_json_path, output_json_path, model):
#     """
#     Đọc dữ liệu từ input_json_path, xử lý và lưu vào output_json_path.
#     """
#     with open(input_json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     new_data = {}

#     for hash_id, content in tqdm(data.items(), desc="Đang xử lý dataset"):
#         new_entry = {}

#         image_path = content.get("image_path", [])
#         image_url = content.get("image_url", [])
        
#         # Trích xuất face
#         faces = extract_faces(image_path,image_url)
#         faces_embbed = []
#         if faces:
#             # Embedding hết face trong faces
#             for face in faces:
#                 face_emb = get_face_embedding(face)
#                 faces_embbed.append(face_emb)
        
#             face_emb_path = os.path.join("/data/npl/ICEK/VACNIC/data/train/faces", f"{hash_id}.npy")
#             np.save(face_emb_path, faces_embbed)
#             new_entry["face_emb_dir"] = face_emb_path
#         else: 
#             new_entry["face_emb_dir"] = []

#         # # Trích xuất objects
#         objects, _ = detect_objects(image_path,image_url)
#         objects_embbed = []
#         if objects:
#             # Embedding hết object trong objects
#             for obj in objects:
#                 object_emb = extract_object_embedding(image_path,image_url, obj)
#                 objects_embbed.append(object_emb)

#             object_emb_path = os.path.join("/data/npl/ICEK/VACNIC/data/train/objects", f"{hash_id}.npy")
#             np.save(object_emb_path, objects_embbed)
#             new_entry["obj_emb_dir"] = object_emb_path
#         else:
#             new_entry["obj_emb_dir"] = []

#         # new_entry["face_emb_dir"] = []
#         # new_entry["obj_emb_dir"] = []
        
#         caption = content.get("caption", [])

#         # Lấy các câu trong context
#         list_sents_byclip = content.get("sents_byclip", [])
#         sents_byclip = ' '.join(list_sents_byclip)
#         # context_text = " ".join(contexts)

#         # Trích xuất thực thể từ context
#         context_entities = extract_entities(sents_byclip, nlp)
#         names_art = context_entities.get("PERSON", [])
#         org_norp_art = context_entities.get("ORGANIZATION", []) + context_entities.get("NORP", [])
#         gpe_loc_art = context_entities.get("GPE", []) + context_entities.get("LOCATION", [])

#         # Trích xuất thực thể từ captions
#         captions_text = caption
#         caption_entities = extract_entities(captions_text, nlp)
#         names_caption = caption_entities.get("PERSON", [])
#         org_norp_caption = caption_entities.get("ORGANIZATION", []) + caption_entities.get("NORP", [])
#         gpe_loc_caption = caption_entities.get("GPE", []) + caption_entities.get("LOCATION", [])

#         # Trích xuất thực thể từ captions và thêm trường ner_cap
#         ner_cap = []
#         for ent_type, entities in caption_entities.items():
#             ner_cap.extend(entities)
#         new_entry["ner_cap"] = list(set(ner_cap)) 

#         # Kết hợp context và caption để tạo named_entities
#         named_entites = []
#         for ent_type, entities in context_entities.items():
#             named_entites.extend(entities)
#         for ent_type, entities in caption_entities.items():
#             named_entites.extend(entities)
#         new_entry["named_entites"] = list(set(named_entites)) 

#         # Tạo các trường mới với loại thực thể duy nhất
#         new_entry["names_art"] = list(set(names_art))
#         new_entry["org_norp_art"] = list(set(org_norp_art))
#         new_entry["gpe_loc_art"] = list(set(gpe_loc_art))
#         new_entry["org_norp_cap"] = list(set(org_norp_caption))
#         new_entry["gpe_loc_cap"] = list(set(gpe_loc_caption))

#         # Kết hợp thực thể từ context và captions
#         new_entry["names"] = list(set(names_art + names_caption))
#         new_entry["org_norp"] = list(set(org_norp_art + org_norp_caption))
#         new_entry["gpe_loc"] = list(set(gpe_loc_art + gpe_loc_caption))
        
#         # Thêm các trường hiện tại
#         new_entry["image_path"] = image_path
#         new_entry["caption"] = caption
#         # new_entry["caption"] = model.word_segment(caption)
#         # new_entry["context"] = contexts

#         name_pos_cap = find_name_positions(caption, names_caption)
#         if not name_pos_cap:
#             name_pos_cap = []
#         new_entry["name_pos_cap"] = name_pos_cap 
        
#         # Thêm trường sents_byclip bằng clip_get_sentences
#         # sentences_by_clip, _ = retrieve_relevant_sentences(image_path, contexts)
#         # sents_byclip = model.word_segment(sents_byclip)
#         new_entry["sents_byclip"] = "\n\n".join(list_sents_byclip)
        
#         new_data[hash_id] = new_entry

#         with open(output_json_path, 'w', encoding='utf-8') as f:
#             json.dump(new_data, f, ensure_ascii=False, indent=4)

#     print(f"Dữ liệu đã được xử lý và lưu vào {output_json_path}")


# if __name__ == "__main__":
#     input_json = r"/data/npl/ICEK/VACNIC/data/train/context_4.json" 
#     output_json = r"/data/npl/ICEK/VACNIC/data/train/train_4_con.json"  
    
#     process_dataset(input_json, output_json, nlp)
 
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

# Define label mapping for vncorenlp NER labels
label_mapping = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORGANIZATION", "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    "GPE": "GPE", "B-GPE": "GPE", "I-GPE": "GPE",
    "NORP": "NORP", "B-NORP": "NORP", "I-NORP": "NORP",
    "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
}

# Initialize vncorenlp
vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir="/data2/npl/ICEK/VnCoreNLP")

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

def extract_entities(text, vncore):
    """
    Extract entities from text using vncorenlp.
    Apply abbreviation check for entities containing periods.
    Return a dictionary with entity types (PERSON, ORGANIZATION, etc.) as lists of unique entities.
    """
    entities = defaultdict(set)
    if not text.strip():
        return {}

    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    for sent in sentences:
        if not sent.strip():
            continue
        try:
            annotated_text = vncore.annotate_text(sent)
            for subsent in annotated_text:
                for word in annotated_text[subsent]:
                    ner_label = word.get('nerLabel', '')
                    if ner_label and ner_label != 'O':
                        ent_type = label_mapping.get(ner_label, '')
                        if ent_type:
                            ent_text = word.get('wordForm', '').strip()
                            ent_text = ' '.join(ent_text.split('_'))
                            if '.' in ent_text:
                                if is_abbreviation(ent_text):
                                    entities[ent_type].add(ent_text)
                            else:
                                entities[ent_type].add(ent_text)
        except Exception as e:
            print(f"Error processing sentence: {e}")
    return {key: list(value) for key, value in entities.items()}

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

        image_path = content.get("image_path", [])
        image_url = content.get("image_url", [])

        # Extract faces
        faces = extract_faces(image_path, image_url)
        faces_embbed = []
        if faces:
            for face in faces:
                face_emb = get_face_embedding(face)
                faces_embbed.append(face_emb)
            face_emb_path = os.path.join("/data/npl/ICEK/VACNIC/data/train/faces", f"{hash_id}.npy")
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
            object_emb_path = os.path.join("/data/npl/ICEK/VACNIC/data/train/objects", f"{hash_id}.npy")
            np.save(object_emb_path, objects_embbed)
            new_entry["obj_emb_dir"] = object_emb_path
        else:
            new_entry["obj_emb_dir"] = []

        caption = content.get("caption", [])
        list_sents_byclip = content.get("sents_byclip", [])
        sents_byclip = ' '.join(list_sents_byclip)

        # Extract entities from context
        context_entities = extract_entities(sents_byclip, vncore)
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
    input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/val.json"
    output_json = r"/data2/npl/ICEK/vacnic/data/val.json"
    process_dataset(input_json, output_json, vncore)
    input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/test.json"
    output_json = r"/data2/npl/ICEK/vacnic/data/test.json"
    process_dataset(input_json, output_json, vncore)
    input_json = r"/data2/npl/ICEK/Wikipedia/content/ver4/train.json"
    output_json = r"/data2/npl/ICEK/vacnic/data/train.json"
    process_dataset(input_json, output_json, vncore)