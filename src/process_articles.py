import os 
from tqdm import tqdm 
import json 
import re 
import py_vncorenlp
from typing import List, Dict, Tuple

# LABEL_MAP = {
#     "PER": "PER", "PERSON": "PER",
#     "ORG": "ORG", "ORGANIZATION": "ORG", "NORP": "ORG",
#     "LOC": "LOC", "GPE": "LOC", "LOCATION": "LOC"

# }
LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORGANIZATION", "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    "GPE": "GPE", "B-GPE": "GPE", "I-GPE": "GPE",
}

def is_abbreviation(entity):
    """
    Kiểm tra xem một chuỗi có phải viết tắt theo dạng A.B. hay không.
    Pattern: A. B. (có thể thêm tên tiếp theo sau)
    """
    pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
    return re.match(pattern, entity) is not None

def align_tokens_to_text(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Given the original text and a list of token strings in order,
    return (start_char, end_char) for each token by scanning forward.

    Robust to repeated tokens and punctuation because we always search from the moving cursor.
    """
    offsets = []
    cursor = 0
    for tok in tokens:
        # Try to find the next occurrence of token from the current cursor
        # Escape regex special chars just in case
        pattern = re.escape(tok)
        m = re.search(pattern, text[cursor:])
        if m:
            start = cursor + m.start()
            end = cursor + m.end()
        else:
            # Fallbacks in rare alignment edge cases
            # 1) Skip any whitespace at cursor and try direct slice
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if text[cursor:cursor+len(tok)] == tok:
                start = cursor
                end = cursor + len(tok)
            else:
                # 2) Last resort: global find from cursor
                idx = text.find(tok, cursor)
                if idx == -1:
                    raise ValueError(f"Cannot align token '{tok}' near index {cursor}")
                start = idx
                end = idx + len(tok)
        offsets.append((start, end))
        cursor = end
    return offsets

def get_entities(doc, article_full):
    entities = []
    tokens = []
    for subsent in doc:
        for word in doc[subsent]:
            ent_type = LABEL_MAP.get(word.get('nerLabel', ''), '')
            ent_text = word.get('wordForm', '').strip()
            if ent_type and ent_text:
                ent_text = (' '.join(ent_text.split('_')).strip("•"))
                tokens.append({
                    "text": ent_text,
                    "label": ent_type,          # luôn là PERON / ORGANIZATION / LOCATION
                })
    token_texts = [t['text'] for t in tokens]
    token_offsets = align_tokens_to_text(article_full, token_texts)
    for t, (s, e) in zip(tokens, token_offsets):
        entities.append({
            'text': t['text'],
            'label': t['label'],  
            "position":[s,e]
        })
    return entities

def make_ner_dict_by_type(processed_doc, ent_list, ent_type_list, article_full):
    # make dict for unique ners with format as: {"Bush": PERSON_1}
    person_count = 1 # total count of PERSON type entities
    org_count = 1 # total count of ORG type entities
    gpe_count = 1 # total count of GPE type entities

    unique_ner_dict = {}
    new_ner_type_list = []

    for i, ent in enumerate(ent_list):
        if ent in unique_ner_dict.keys():
            new_ner_type_list.append(unique_ner_dict[ent])
        
        elif ent_type_list[i] == "PERSON" or ent_type_list[i] == "PER":
            ner_type = "<PERSON>_" + f"{person_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            person_count += 1
        elif ent_type_list[i] in ["ORGANIZATION", "ORG", "NORP"]:
            ner_type = "<ORGANIZATION>_" + f"{org_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            org_count += 1
        elif ent_type_list[i] in ["GPE", "LOC","LOCATION"]:
            ner_type = "<LOCATION>_" + f"{gpe_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            gpe_count += 1
        
    entities_type = {} # dict with ner labels replaced by "PERSON_i", "ORG_j", "GPE_k"

    entities = get_entities(processed_doc, article_full)

    for i, ent in enumerate(entities):
        entities_type[i] = ent
        entities_type[i]["label"] = new_ner_type_list[i]
    # print(entities_type)

    start_pos_list = [sample["position"][0] for sample in entities_type.values()] # list of start positions for each entity
    # print(start_pos_list)
        
    return entities_type, start_pos_list, person_count, org_count, gpe_count




def save_full_processed_articles_all_ent_by_count(
        data_dict: dict,
        out_dir: str,
        tokenizer,
        nlp              
    ):
    """
    Sinh file {hash_id}.json chứa 'input_ids' (article đã ẩn danh NER).
    Nếu article text đã nằm trong JSON (trường 'sents_byclip') sẽ đọc trực tiếp,
    ngược lại sẽ mở file <article_full_text_dir>/<hash_id>.txt (tuỳ chọn).
    """
    os.makedirs(out_dir, exist_ok=True)
    # i = 0
    for key, meta in tqdm(data_dict.items(), desc="make NER masks"):
        # if(i%100==0):
        #     print(f"[PROCESSING] DONE {i}")
        if meta.get("sents_byclip"):              
            article_full = meta["sents_byclip"].replace('\n\n',' ')
        else:
            raise ValueError(f"{key} thiếu 'sents_byclip'; "
                             "hãy thêm hoặc truyền đường dẫn txt.")
        sentences = re.split(r'(?<=[\.!?])\s+', article_full.strip())
        if not sentences:
            return {}
        processed_text=[]
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            try:
                annotated_text = nlp.annotate_text(sent)
                processed_text.append(annotated_text)
            except Exception as e:
                print(f"Lỗi khi annotating text: {e}")
        
        entities      = get_entities(processed_text, article_full)  

        ent_list      = [e["text"]   for e in entities]
        ent_type_list = [e["label"]  for e in entities]

        # 3. Tạo mapping <PERSON>_1 … và id list
        entities_type, start_pos_list, *_ = \
            make_ner_dict_by_type(processed_text, ent_list, ent_type_list, article_full)

        ent_len_list = [len(tokenizer(t)["input_ids"]) - 2 for t in ent_list]

        article_ids_ner = make_new_article_ids_all_ent(
                              article_full, ent_list, entities_type, tokenizer)

        # 4. Ghi file JSON (chỉ ghi nếu chưa tồn tại)
        out_path = os.path.join(out_dir, f"{key}.json")
        if not os.path.isfile(out_path):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(article_ids_ner, f, ensure_ascii=False)
                # print("đã lưu nha bro")
        # i+=1


def make_new_article_ids_all_ent(article_full, ent_list, entities_type, tokenizer):
    article_ids_ner = tokenizer(article_full)["input_ids"]
    # article_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:

        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        idx = find_first_sublist(article_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # print(ent_ids_original)
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            # print(tokenizer(ner_chain)["input_ids"][1:-1])
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # print(ent_ids_original)
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        counter += 1

    return {"input_ids":article_ids_ner}


def replace_sublist(seq, sublist, replacement):
    length = len(replacement)
    index = 0
    for start, end in iter(lambda: find_first_sublist(seq, sublist, index), None):
        seq[start:end] = replacement
        index = start + length
    return seq

def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index:index+length] == sublist:
            return index, index+length


def get_caption_with_ent_type(nlp, caption, tokenizer):
    processed_doc = nlp.annotate_text(caption)
    entities = get_entities(processed_doc, caption)
        
    ent_list = [ entities[i]["text"] for i in range(len(entities)) ]
    ent_type_list = [ entities[i]["label"] for i in range(len(entities)) ]
        
    entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(processed_doc, ent_list, ent_type_list, caption)

    new_caption, caption_ids_ner = make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer)
    return new_caption, caption_ids_ner


def make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer):
    caption_ids_ner = tokenizer(caption)["input_ids"]
    # caption_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # in case entities were in the middle of the sentence
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # if ent_ids_original in caption_ids_ner:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        counter += 1
    return tokenizer.decode(caption_ids_ner), caption_ids_ner


def get_person_ids_position(article_ids_replaced, person_token_id=None, article_max_length=512, is_tgt_input=False):
    position_list = []
    i = 0
    while i < len(article_ids_replaced):
        position_i = []
        if article_ids_replaced[i] == person_token_id and i < article_max_length:
            if is_tgt_input:
                position_i.append(i+1)
            else:
                position_i.append(i)
            for j in range(i, len(article_ids_replaced)):
                if article_ids_replaced[j] == person_token_id:
                    continue
                else:
                    if is_tgt_input:
                        position_i.append(j)
                    else:
                        position_i.append(j-1)
                    i=j-1
                    # print(i)
                    break
            position_list.append(position_i)
            # print("i:",i)
        i += 1
    return position_list


def add_name_pos_list_to_dict(data_dict, nlp, tokenizer):
    new_dict = {}
    for key, value in tqdm(data_dict.items()):
        new_dict[key] = {}
        new_dict[key] = value
        _, caption_ids_ner = get_caption_with_ent_type(nlp, value["caption"], tokenizer)
        position_list = get_person_ids_position(caption_ids_ner, person_token_id=PERSON_ID, article_max_length=20, is_tgt_input=True)

        new_dict[key]["name_pos_cap"] = position_list
    return new_dict


if __name__ == "__main__":

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data2/npl/ICEK/vacnic/bartpho-syllable-base")
    tokenizer.add_special_tokens({"additional_special_tokens":["<PERSON>", "<ORGANIZATION>", "<LOCATION>"]})
   
    PERSON_ID = tokenizer.convert_tokens_to_ids('<PERSON>')

    py_vncorenlp.download_model(save_dir="/data2/npl/ICEK/VnCoreNLP")
    nlp = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "pos", "ner"],
        save_dir="/data2/npl/ICEK/VnCoreNLP",
        max_heap_size='-Xmx10g'
    )
    # nlp = stanza.Pipeline(lang='vi', processors='tokenize,ner')

    with open('/data2/npl/ICEK/vacnic/data/test.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)
    print("[DEBUG] DATA LOADED, PROCESSING")
    OUT_DIR = "/data2/npl/ICEK/vacnic/data/embeddings/article_all_ent_by_count_dir/test"
    save_full_processed_articles_all_ent_by_count(
            data_dict=data_dict,
            out_dir=OUT_DIR,
            tokenizer=tokenizer,
            nlp=nlp)

    with open('/data2/npl/ICEK/vacnic/data/val.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)
    print("[DEBUG] DATA LOADED, PROCESSING")
    OUT_DIR = "/data2/npl/ICEK/vacnic/data/embeddings/article_all_ent_by_count_dir/val"
    save_full_processed_articles_all_ent_by_count(
            data_dict=data_dict,
            out_dir=OUT_DIR,
            tokenizer=tokenizer,
            nlp=nlp)

    with open('/data2/npl/ICEK/vacnic/data/train.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)
    print("[DEBUG] DATA LOADED, PROCESSING")
    OUT_DIR = "/data2/npl/ICEK/vacnic/data/embeddings/article_all_ent_by_count_dir/train"
    save_full_processed_articles_all_ent_by_count(
            data_dict=data_dict,
            out_dir=OUT_DIR,
            tokenizer=tokenizer,
            nlp=nlp)


