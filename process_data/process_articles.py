import os 
from tqdm import tqdm 
import json 
import re 
from typing import List, Tuple
from vncore_singleton import get_vncore
from utils_precompute import extract_entities

LABEL_MAP = {
    "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
    "ORG": "ORGANIZATION", "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
    "LOC": "LOCATION", "B-LOC": "LOCATION", "I-LOC": "LOCATION",
    # "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
}
def align_tokens_to_text(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Accent-insensitive, whitespace-tolerant alignment.
    Returns (start_char, end_char) in ORIGINAL text for each token.
    """
    offsets: List[Tuple[int, int]] = []
    n_cursor = 0  
    for tok in tokens:
        try:
            m = re.search(tok, text[n_cursor:])
        except:
            print(f"[ERR] error searching:text = {text}\n \n tokens:{tokens}")
        if m is None:
            print(f"[WARN] invalid tok = {tok} \n \n text = {text}\n \n tokens:{tokens}")
            offsets.append((-1, -1))
            continue
        start_index = (n_cursor + m.start()) if m.re is not None and m.string is text else m.start()
        end_index   = (n_cursor + m.end())   if m.re is not None and m.string is text else m.end()

        offsets.append((start_index, end_index))
        n_cursor = end_index

    return offsets

def _has_alnum(s: str) -> bool:
    # keep tokens that have at least one letter/number (works with Unicode)
    return any(ch.isalnum() for ch in s)

def get_entities(doc, article_full):
    entities = []
    tokens = []
    for sent in doc:
        for subsent in sent:
            for word in sent[subsent]:
                ent_type = LABEL_MAP.get(word.get('nerLabel', ''), '')
                ent_text = word.get('wordForm', '').strip()
                if ent_type and ent_text:
                    raw_ent_text = (' '.join(ent_text.split('_')).strip("•"))
                    ent_text = re.sub(r"[()\[\]{}'\"“”‘’]", "", raw_ent_text).strip()
                    if not ent_type or not ent_text or not _has_alnum(ent_text):
                        continue
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
    ent_count = 1

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
            ner_type = "<PERSON>_" + f"{org_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            org_count += 1
        elif ent_type_list[i] in ["GPE", "LOC","LOCATION"]:
            ner_type = "<GPELOC>_" + f"{gpe_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            gpe_count += 1
        else:
            ner_type = "<ENT>_" + f"{gpe_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            ent_count += 1

        # elif ent_type_list[i] in ["MISC"]:
        #     ner_type = "<MISC>_" + f"{gpe_count}"
        #     unique_ner_dict[ent] = ner_type
        #     new_ner_type_list.append(ner_type)
        #     gpe_count += 1
        
    entities_type = {} # dict with ner labels replaced by "PERSON_i", "ORG_j", "GPE_k"

    entities = get_entities(processed_doc, article_full)
    for i, ent in enumerate(entities):
        entities_type[i] = ent
        entities_type[i]["label"] = new_ner_type_list[i]
    # print(entities_type)

    start_pos_list = [sample["position"][0] for sample in entities_type.values()] # list of start positions for each entity
    # print(start_pos_list)
        
    return entities_type, start_pos_list, person_count, org_count, gpe_count, ent_count




def save_full_processed_articles_all_ent_by_count(
        data_dict: dict,
        out_dir: str,
        tokenizer,
        nlp              
    ):
    """
    Sinh file {hash_id}.json chứa 'input_ids' (article đã ẩn danh NER).
    Nếu article text đã nằm trong JSON (trường 'paragraphs') sẽ đọc trực tiếp,
    ngược lại sẽ mở file <article_full_text_dir>/<hash_id>.txt (tuỳ chọn).
    """
    os.makedirs(out_dir, exist_ok=True)
    # i = 0
    for key, meta in tqdm(data_dict.items(), desc="making 'articles_all_ent_by_count'"):
        # if(i%100==0):
        #     print(f"[PROCESSING] DONE {i}")
        if meta.get("paragraphs"):              
            article_full = ". ".join(meta["paragraphs"])
        else:
            raise ValueError(f"{key} thiếu 'paragraphs'; "
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
                # print(f"[DEBUG] annotated_text: {annotated_text}")
                processed_text.append(annotated_text)
            except Exception as e:
                print(f"Lỗi khi annotating text: {e}")
        
        entities      = get_entities(processed_text, article_full)  

        ent_list      = [e["text"]   for e in entities]
        ent_type_list = [e["label"]  for e in entities]

        # 3. Tạo mapping <PERSON>_1 … và id list
        entities_type, start_pos_list, *_ = \
            make_ner_dict_by_type(processed_text, ent_list, ent_type_list, article_full)

        # print(f"[DEBUG] ent_list {ent_list} \n ent_type_list{ent_type_list} \n entities_type {entities_type}")
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
    sentences = caption
    if not sentences:
        return {}
    processed_text=[]
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            annotated_text = nlp.annotate_text(sent)
            # print(f"[DEBUG] annotated_text: {annotated_text}")
            processed_text.append(annotated_text)
        except Exception as e:
            print(f"Lỗi khi annotating text: {e}")
    entities = get_entities(processed_text, caption)
        
    ent_list = [ entities[i]["text"] for i in range(len(entities)) ]
    ent_type_list = [ entities[i]["label"] for i in range(len(entities)) ]
        
    entities_type, start_pos_list, *_ = make_ner_dict_by_type(processed_text, ent_list, ent_type_list, caption)

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
    for key, value in tqdm(data_dict.items(), desc="adding 'name_pos_cap' into processed dict:"):
        new_dict[key] = {}
        new_dict[key] = value
        _, caption_ids_ner = get_caption_with_ent_type(nlp, value["caption"], tokenizer)
        position_list = get_person_ids_position(caption_ids_ner, person_token_id=PERSON_ID, article_max_length=20, is_tgt_input=True)

        new_dict[key]["name_pos_cap"] = position_list
    return new_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/datastore/npl/ICEK/vacnic/vacnic_pretrained_model/bartpho-syllable")
    tokenizer.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
    print("[DEBUG] Tokenizer loaded")
   
    PERSON_ID = tokenizer.convert_tokens_to_ids('<PERSON>')

    nlp = get_vncore(r"/datastore/npl/ICEK/VnCoreNLP", with_heap=True)

    with open(r'/datastore/npl/ICEK/Wikipedia/content/ver5/demo20.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)
    print("[DEBUG] DATA LOADED, PROCESSING")
    OUT_DIR = r"/datastore/npl/ICEK/vacnic/data/embedding/article_all_ent_by_count_dir/demo20"
    save_full_processed_articles_all_ent_by_count(
            data_dict=data_dict,
            out_dir=OUT_DIR, 
            tokenizer=tokenizer,
            nlp=nlp)

    with open (r'/datastore/npl/ICEK/vacnic/data/demo20.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)
    new_data_dict =  add_name_pos_list_to_dict(data_dict, nlp, tokenizer)
    with open(r'/datastore/npl/ICEK/vacnic/data/demo20.json', 'w', encoding='utf-8') as f:
        json.dump(new_data_dict, f, ensure_ascii=False, indent=4)

    # with open(r'/datastore/npl/ICEK/Wikipedia/content/ver5/test.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # print("[DEBUG] DATA LOADED, PROCESSING")
    # OUT_DIR = r"/datastore/npl/ICEK/vacnic/data/embedding/article_all_ent_by_count_dir/test"
    # save_full_processed_articles_all_ent_by_count(
    #         data_dict=data_dict,
    #         out_dir=OUT_DIR,
    #         tokenizer=tokenizer,
    #         nlp=nlp)
    
    # with open (r'/datastore/npl/ICEK/vacnic/data/test.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # new_data_dict =  add_name_pos_list_to_dict(data_dict, nlp, tokenizer)
    # with open(r'/datastore/npl/ICEK/vacnic/data/test.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_data_dict, f, ensure_ascii=False, indent=4)
    
    # with open(r'/datastore/npl/ICEK/Wikipedia/content/ver5/val.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # print("[DEBUG] DATA LOADED, PROCESSING")
    # OUT_DIR = r"/datastore/npl/ICEK/vacnic/data/embedding/article_all_ent_by_count_dir/val"
    # save_full_processed_articles_all_ent_by_count(
    #         data_dict=data_dict,
    #         out_dir=OUT_DIR,
    #         tokenizer=tokenizer,
    #         nlp=nlp)
    
    # with open (r'/datastore/npl/ICEK/vacnic/data/val.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # new_data_dict =  add_name_pos_list_to_dict(data_dict, nlp, tokenizer)
    # with open(r'/datastore/npl/ICEK/vacnic/data/val.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_data_dict, f, ensure_ascii=False, indent=4)

    # with open(r'/datastore/npl/ICEK/Wikipedia/content/ver5/train.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # print("[DEBUG] DATA LOADED, PROCESSING")
    # OUT_DIR = r"/datastore/npl/ICEK/vacnic/data/embedding/article_all_ent_by_count_dir/train"
    # save_full_processed_articles_all_ent_by_count(
    #         data_dict=data_dict,
    #         out_dir=OUT_DIR,
    #         tokenizer=tokenizer,
    #         nlp=nlp)
    
    # with open (r'/datastore/npl/ICEK/vacnic/data/train.json','r',encoding='utf-8') as f:
    #     data_dict = json.load(f)
    # new_data_dict =  add_name_pos_list_to_dict(data_dict, nlp, tokenizer)
    # with open(r'/datastore/npl/ICEK/vacnic/data/train.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_data_dict, f, ensure_ascii=False, indent=4)

