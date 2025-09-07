import os
import json
import copy
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

# NOTE: We keep BartTokenizer defaults to preserve BOS/PAD/EOS ids (0/1/2)
# and the additional_special_tokens indexing used elsewhere in the repo.
from transformers import BartTokenizer

# ------------------------------
# Collate + padding utilities
# ------------------------------

def get_max_len_list(seq_list_of_list):
    max_len_list = []
    for seq_list in seq_list_of_list:
        max_len_list.extend([len(seq) for seq in seq_list])
    return max(max_len_list) if len(max_len_list) > 0 else 1


def get_max_len(seq_tensor_list_of_list):
    max_len_list = []
    for seq_tensor_list in seq_tensor_list_of_list:
        max_len_list.append(max([seq.size(1) for seq in seq_tensor_list]))
    return max(max_len_list) if len(max_len_list) > 0 else 1


def pad_sequence(seq_tensor_list, pad_token_id, max_len=None):
    if len(seq_tensor_list) == 0:
        return torch.empty(0)
    if max_len is None:
        max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    for seq in seq_tensor_list:
        pad_num = max_len - seq.size(1)
        if pad_num > 0:
            to_be_padded = torch.tensor([pad_token]*pad_num, dtype=torch.long).unsqueeze(0)
            padded_list.append(torch.cat((seq, to_be_padded), dim=1))
        else:
            padded_list.append(seq)
    return torch.stack(padded_list)


def pad_sequence_from_list(seq_list_list, special_token_id, bos_token_id, pad_token_id, eos_token_id, max_len):
    max_num_seq = max([len(seq_list) for seq_list in seq_list_list]) if len(seq_list_list) > 0 else 1
    padded_list_all = []
    for seq_list in seq_list_list:
        padded_list = []
        for seq in seq_list:
            pad_num = max_len - len(seq)
            seq = seq + [pad_token_id] * pad_num
            if max_num_seq == 1:
                padded_list.append([seq])
            else:
                padded_list.append(seq)
        if len(seq_list) < max_num_seq:
            pad_batch_wise = [bos_token_id] + [special_token_id] + [eos_token_id] + [pad_token_id] * (max_len-3)
            for _ in range(max_num_seq - len(seq_list)):
                padded_list.append(pad_batch_wise)
        padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
    return torch.stack(padded_list_all)


def pad_tensor_feat(feat_np_list, pad_feat_tensor):
    len_list = []
    for feat in feat_np_list:
        if feat.shape[1] == 0:
            len_list.append(0)
        else:
            len_list.append(feat.shape[0])
    max_len = max(len_list) if len_list else 0
    padded_list = []
    for i, feat in enumerate(feat_np_list):
        pad_num = max_len - len_list[i]
        if pad_num > 0:
            to_be_padded = pad_num * [pad_feat_tensor]
            to_be_padded = torch.stack(to_be_padded, dim=0).squeeze(1)
            if feat.shape[1] != 0:
                padded_list.append(torch.cat((torch.from_numpy(feat), to_be_padded), dim=0).squeeze(1))
            else:
                padded_list.append(to_be_padded)
        elif max_len == 0:
            to_be_padded = 1 * [pad_feat_tensor]
            to_be_padded = torch.stack(to_be_padded, dim=0).squeeze(1)
            padded_list.append(to_be_padded)
        else:
            padded_list.append(torch.from_numpy(feat))
    return torch.stack(padded_list) if len(padded_list) > 0 else torch.empty(0)


def get_person_ids_position(article_ids_replaced, person_token_id=50265, article_max_length=512, is_tgt_input=False):
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
                    i = j-1
                    break
            position_list.append(position_i)
        i += 1
    return position_list


def concat_ner(ner_list, entity_token_start, entity_token_end):
    concat_ner_list = []
    if entity_token_start == "no" or entity_token_end == "no":
        for ner in ner_list:
            concat_ner_list.extend(ner)
    elif entity_token_start == "|":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start])
            else:
                concat_ner_list.extend([ner])
    elif entity_token_start == "</s>":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start + entity_token_end])
            else:
                concat_ner_list.extend([ner])
    else:
        for ner in ner_list:
            concat_ner_list.extend([entity_token_start + " " + ner + " " + entity_token_end])
    return " ".join(concat_ner_list)


def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index:index+length] == sublist:
            return index, index+length


def replace_sublist(seq, sublist, replacement):
    length = len(replacement)
    index = 0
    for start, end in iter(lambda: find_first_sublist(seq, sublist, index), None):
        seq[start:end] = replacement
        index = start + length
    return seq


# ------------------------------
# Vietnamese NER via VnCoreNLP (optional)
# ------------------------------

class _VNCoreNER:
    def __init__(self, save_dir: str = "./vncorenlp", jar_name: str = "VnCoreNLP-1.1.1.jar"):
        self.ok = False
        try:
            from vncorenlp import VnCoreNLP  # type: ignore
            os.makedirs(save_dir, exist_ok=True)
            # This will auto-download if missing
            self.rdr = VnCoreNLP(os.path.join(save_dir, jar_name),
                                  save_dir=save_dir,
                                  annotators="wseg,pos,ner",
                                  max_heap_size="-Xmx8g")
            self.ok = True
        except Exception:
            self.rdr = None
            self.ok = False

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Return dict with keys: PER, ORG, LOC based on NER results.
        If unavailable, returns empty lists.
        """
        out = {"PER": [], "ORG": [], "LOC": []}
        if not self.ok or not text:
            return out
        try:
            anns = self.rdr.annotate(text)
            # anns is list of sentences; each token has "nerLabel" and "form"
            seen = {"PER": set(), "ORG": set(), "LOC": set()}
            for sent in anns:
                chunk = []
                label = None
                for tok in sent:
                    ner = tok.get("nerLabel", "O")
                    form = tok.get("form", "")
                    if ner.startswith("B-"):
                        if chunk and label:
                            val = " ".join(chunk).strip()
                            if val and val not in seen[label]:
                                out[label].append(val)
                                seen[label].add(val)
                        chunk = [form]
                        label = ner.split("-", 1)[1]
                    elif ner.startswith("I-") and label:
                        chunk.append(form)
                    else:
                        if chunk and label:
                            val = " ".join(chunk).strip()
                            if val and val not in seen[label]:
                                out[label].append(val)
                                seen[label].add(val)
                        chunk = []
                        label = None
                if chunk and label:
                    val = " ".join(chunk).strip()
                    if val and val not in seen[label]:
                        out[label].append(val)
                        seen[label].add(val)
        except Exception:
            pass
        return out


# ------------------------------
# Entity helpers for IDs
# ------------------------------

def make_new_entity_ids(caption: str, ent_list: List[str], tokenizer: BartTokenizer, ent_separator: str = "<ent>", max_length: int = 80):
    caption_ids_ner = tokenizer(caption, add_special_tokens=False)["input_ids"]
    sep_token = tokenizer(ent_separator, add_special_tokens=False)["input_ids"]
    noname_token = tokenizer("<NONAME>")["input_ids"][1:-1]

    ent_ids_flatten: List[int] = []
    ent_ids_separate: List[List[int]] = []

    for ent in ent_list:
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            ent_ids_flatten.extend(ent_ids_original)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break
        else:
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ent_ids_flatten.extend(ent_ids_original_start)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original_start + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break

    if len(ent_ids_flatten) == 0:
        ent_ids_flatten.extend(noname_token)

    ent_ids_flatten = [tokenizer.bos_token_id] + ent_ids_flatten + [tokenizer.eos_token_id]
    if len(ent_ids_flatten) < max_length:
        ent_ids_flatten = ent_ids_flatten + [tokenizer.pad_token_id] * (max_length - len(ent_ids_flatten))

    ent_ids_separate.append([tokenizer.bos_token_id] + noname_token + [tokenizer.eos_token_id])
    ent_ids_separate = pad_list(ent_ids_separate, tokenizer.pad_token_id)
    return torch.LongTensor([ent_ids_flatten]), ent_ids_separate


def pad_list(list_of_name_ids: List[List[int]], pad_token: int):
    max_len = max([len(seq) for seq in list_of_name_ids]) if list_of_name_ids else 1
    padded_list = []
    for seq in list_of_name_ids:
        if len(seq) == max_len:
            padded_list.append(seq)
        else:
            padded_num = max_len - len(seq)
            seq = seq + [pad_token] * padded_num
            padded_list.append(seq)
    return padded_list


def make_new_article_ids_all_ent(article_full: str, ent_list: List[str], ent_types: List[str], tokenizer: BartTokenizer):
    """Replace entities with their generic tags while preserving token count length-by-length.
    ent_types must be aligned with ent_list and contain values in {"PERSON","ORG","LOC"}.
    """
    article_ids_ner = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        ent_type = ent_types[counter]
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        idx = find_first_sublist(article_ids_ner, ent_ids_original, start=0)
        if idx is None:
            ent_ids_original = tokenizer(f"{ent}")["input_ids"][1:-1]
        label_root = "<PERSON>" if ent_type == "PERSON" else ("<ORGNORP>" if ent_type == "ORG" else "<GPELOC>")
        ner_chain = " ".join([label_root] * len(ent_ids_original)) if len(ent_ids_original) > 0 else label_root
        article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1] or [tokenizer.pad_token_id])
        counter += 1
    return {"input_ids": article_ids_ner}


# ------------------------------
# Collate function (renamed to viwiki but keep GoodNews alias for compatibility)
# ------------------------------

def collate_fn_viwiki_entity_type(batch):
    article_list, article_id_list, article_ner_mask_id_list = [], [], []
    caption_list, caption_id_list, caption_id_clip_list = [], [], []

    names_art_list, names_art_ids_list = [], []
    org_norp_gpe_loc_art_list, org_norp_gpe_loc_art_ids_list = [], []

    names_list, names_ids_list = [], []
    org_norp_gpe_loc_list, org_norp_gpe_loc_ids_list = [], []

    names_ids_flatten_list, org_norp_gpe_loc_ids_flatten_list = [], []

    all_gt_ner_list, all_gt_ner_ids_list = [], []
    face_emb_list, obj_emb_list, img_tensor_list = [], [], []

    face_pad = torch.ones((1, 512))
    obj_pad = torch.ones((1, 2048))

    person_id_positions_list, person_id_positions_cap_list = [], []

    for sample in batch:
        article_list.append(sample["article"]) 
        article_id_list.append(sample["article_ids"]) 
        article_ner_mask_id_list.append(sample["article_ner_mask_ids"]) 

        caption_list.append(sample["caption"]) 
        caption_id_list.append(sample["caption_ids"]) 
        if sample.get("caption_ids_clip") is not None:
            caption_id_clip_list.append(sample["caption_ids_clip"]) 

        names_art_list.append(sample["names_art"]) 
        names_art_ids_list.append(sample["names_art_ids"]) 
        org_norp_gpe_loc_art_list.append(sample["org_norp_gpe_loc_art"]) 
        org_norp_gpe_loc_art_ids_list.append(sample["org_norp_gpe_loc_art_ids"]) 

        names_list.append(sample["names"]) 
        names_ids_list.append(sample["names_ids"]) 
        org_norp_gpe_loc_list.append(sample["org_norp_gpe_loc"]) 
        org_norp_gpe_loc_ids_list.append(sample["org_norp_gpe_loc_ids"]) 

        all_gt_ner_list.append(sample["all_gt_ner"]) 
        all_gt_ner_ids_list.append(sample["all_gt_ner_ids"]) 

        face_emb_list.append(sample["face_emb"]) 
        obj_emb_list.append(sample["obj_emb"]) 
        img_tensor_list.append(sample["img_tensor"]) 

        person_id_positions_list.append(sample["person_id_positions"]) 
        person_id_positions_cap_list.append(sample["person_id_positions_cap"]) 

        names_ids_flatten_list.append(sample["names_ids_flatten"]) 
        org_norp_gpe_loc_ids_flatten_list.append(sample["org_norp_gpe_loc_ids_flatten"]) 

    max_len_input = get_max_len([article_id_list, article_ner_mask_id_list])
    article_ids_batch = pad_sequence(article_id_list, 1, max_len=max_len_input)
    article_ner_mask_ids_batch = pad_sequence(article_ner_mask_id_list, 1, max_len=max_len_input)

    caption_ids_batch = pad_sequence(caption_id_list, 1)
    if len(caption_id_clip_list) > 0:
        caption_ids_clip_batch = pad_sequence(caption_id_clip_list, 0)
    else:
        caption_ids_clip_batch = torch.empty((1, 1))

    max_len_art_ids = get_max_len([names_art_ids_list, org_norp_gpe_loc_art_ids_list])

    max_len_name_ids = get_max_len_list(names_ids_list)
    max_len_org_norp_gpe_loc_ids = get_max_len_list(org_norp_gpe_loc_ids_list)

    names_art_ids_batch = pad_sequence(names_art_ids_list, 1, max_len=max_len_art_ids)
    org_norp_gpe_loc_art_ids_batch = pad_sequence(org_norp_gpe_loc_art_ids_list, 1, max_len=max_len_art_ids)

    # IMPORTANT: these IDs assume BartTokenizer with additional_special_tokens where <NONAME> id=50266
    names_ids_batch = pad_sequence_from_list(names_ids_list, special_token_id=50266, bos_token_id=0, pad_token_id=1, eos_token_id=2, max_len=max_len_name_ids)
    org_norp_gpe_loc_ids_batch = pad_sequence_from_list(org_norp_gpe_loc_ids_list, special_token_id=50266, bos_token_id=0, pad_token_id=1, eos_token_id=2, max_len=max_len_org_norp_gpe_loc_ids)

    all_gt_ner_ids_batch = pad_sequence(all_gt_ner_ids_list, 1)

    max_len_ids_flatten = get_max_len([names_ids_flatten_list, org_norp_gpe_loc_ids_flatten_list])
    names_ids_flatten_batch = pad_sequence(names_ids_flatten_list, 1, max_len=max_len_ids_flatten)
    org_norp_gpe_loc_ids_flatten_batch = pad_sequence(org_norp_gpe_loc_ids_flatten_list, 1, max_len=max_len_ids_flatten)

    img_batch = torch.stack(img_tensor_list, dim=0).squeeze(1)
    face_batch = pad_tensor_feat(face_emb_list, face_pad)
    obj_batch = pad_tensor_feat(obj_emb_list, obj_pad)

    return {
        "article": article_list,
        "article_ids": article_ids_batch.squeeze(1),
        "article_ner_mask_ids": article_ner_mask_ids_batch.squeeze(1),
        "caption": caption_list,
        "caption_ids": caption_ids_batch.squeeze(1),
        "caption_ids_clip": caption_ids_clip_batch.squeeze(1),
        "names_art": names_art_list,
        "names_art_ids": names_art_ids_batch.squeeze(1),
        "org_norp_gpe_loc_art": org_norp_gpe_loc_art_list,
        "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids_batch.squeeze(1),
        "names": names_list,
        "names_ids": names_ids_batch.squeeze(1),
        "org_norp_gpe_loc": org_norp_gpe_loc_list,
        "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids_batch.squeeze(1),
        "all_gt_ner_ids": all_gt_ner_ids_batch.squeeze(1),
        "all_gt_ner": all_gt_ner_list,
        "face_emb": face_batch.float(),
        "obj_emb": obj_batch.float(),
        "img_tensor": img_batch,
        "person_id_positions": person_id_positions_list,
        "person_id_positions_cap": person_id_positions_cap_list,
        "names_ids_flatten": names_ids_flatten_batch.squeeze(1),
        "org_norp_gpe_loc_ids_flatten": org_norp_gpe_loc_ids_flatten_batch.squeeze(1),
    }


# Backward-compatible alias
collate_fn_goodnews_entity_type = collate_fn_viwiki_entity_type


# ------------------------------
# Dataset for ViWiki (demo.json format)
# ------------------------------

class ViWikiDictDatasetEntityTypeFixLenEntPos(Dataset):
    """
    Expect each sample in `data_dict` to follow demo.json schema:
      {
        "image_path": "/path/or/DATADIR/.../img.jpg",
        "paragraphs": [str, ...],
        "scores": [float, ...],            # same length as paragraphs
        "caption": str,                    # short caption/title
        "context": [str, ...]              # optional extra sentences
      }

    This loader auto-extracts Vietnamese NER (PER/ORG/LOC) using VnCoreNLP if available.
    It constructs:
      - article: concatenation of top-k paragraphs by score (k=5 by default).
      - entity lists for article and caption.
      - article_ner_mask_ids: article token ids with entities replaced by <PERSON>/<ORGNORP>/<GPELOC> preserving token counts.
    """

    def __init__(
        self,
        data_dict: Dict[str, Any],
        data_base_dir: str,  # used only if image_path starts with DATADIR/
        tokenizer: BartTokenizer,
        use_clip_tokenizer: bool = False,
        entity_token_start: str = "no",
        entity_token_end: str = "no",
        transform=None,
        max_article_len: int = 512,
        max_ner_type_len: int = 80,
        max_ner_type_len_gt: int = 20,
        retrieved_sent: bool = False,
        person_token_id: int = 50265,
        topk_paragraphs: int = 5,
        vncore_dir: Optional[str] = None,
        vncore_jar: str = "VnCoreNLP-1.1.1.jar",
    ):
        super().__init__()
        self.data_dict = copy.deepcopy(data_dict)
        self.tokenizer = tokenizer
        self.use_clip_tokenizer = use_clip_tokenizer
        self.max_len = max_article_len
        self.transform = transform or Compose([
            Resize(256), CenterCrop(224), ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.entity_token_start = entity_token_start
        self.entity_token_end = entity_token_end
        self.hash_ids = [*data_dict.keys()]
        self.max_ner_type_len = max_ner_type_len
        self.max_ner_type_len_gt = max_ner_type_len_gt
        self.retrieved_sent = retrieved_sent
        self.person_token_id = person_token_id
        self.data_base_dir = data_base_dir
        self.topk = max(1, topk_paragraphs)

        # NER
        self.vnner = _VNCoreNER(save_dir=vncore_dir or "./vncorenlp", jar_name=vncore_jar)

    def _resolve_image_path(self, p: str) -> str:
        if p.startswith("DATADIR/"):
            return os.path.join(self.data_base_dir, p.split("DATADIR/", 1)[1])
        return p

    def _build_article(self, entry: Dict[str, Any]) -> str:
        paras: List[str] = entry.get("paragraphs", [])
        scores: List[float] = entry.get("scores", [])
        if not paras:
            return ""
        if scores and len(scores) == len(paras):
            ranked = sorted(zip(paras, scores), key=lambda x: x[1], reverse=True)
            chosen = [p for p, _ in ranked[:self.topk]]
        else:
            chosen = paras[:self.topk]
        # Add a bit of context, if any
        ctx = entry.get("context", [])
        if ctx:
            chosen = chosen + ctx[:2]
        return " \n".join([s.strip() for s in chosen if s and s.strip()])

    def _extract_entities(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        ner = self.vnner.extract(text)
        names = ner.get("PER", [])
        orgs = ner.get("ORG", [])
        gpe_loc = ner.get("LOC", [])
        # deduplicate while preserving order
        def uniq(seq):
            seen = set(); out = []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        return uniq(names), uniq(orgs), uniq(gpe_loc)

    def __getitem__(self, index: int):
        key = self.hash_ids[index]
        entry = self.data_dict[key]

        img_path = self._resolve_image_path(entry.get("image_path", ""))
        img = Image.open(img_path).convert('RGB')

        article = self._build_article(entry)
        caption = entry.get("caption", "").strip()

        # Entities from article & caption
        names_art, org_art, gpe_loc_art = self._extract_entities(article)
        names_cap, org_cap, gpe_loc_cap = self._extract_entities(caption)
        org_norp_gpe_loc_art = [*org_art, *gpe_loc_art]
        org_norp_gpe_loc_cap = [*org_cap, *gpe_loc_cap]

        all_gt_ner = [*names_cap, *org_cap, *gpe_loc_cap]
        concat_gt_ner = concat_ner(all_gt_ner, self.entity_token_start, self.entity_token_end)
        gt_ner_ids = self.tokenizer(concat_gt_ner, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_ner_type_len_gt)["input_ids"]

        # Tokenized article + masked article ids
        article_ids = self.tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=self.max_len)["input_ids"]
        # Prepare inputs for masked ids
        ent_list = [*names_art, *org_art, *gpe_loc_art]
        ent_types = [*(["PERSON"]*len(names_art)), *(["ORG"]*len(org_art)), *(["LOC"]*len(gpe_loc_art))]
        article_ner_mask_dict = make_new_article_ids_all_ent(article, ent_list, ent_types, self.tokenizer)
        article_ner_mask_ids = torch.LongTensor(article_ner_mask_dict["input_ids"][:self.max_len-1] + [2]) if len(article_ner_mask_dict["input_ids"]) > self.max_len else torch.LongTensor(article_ner_mask_dict["input_ids"])
        article_ner_mask_ids = article_ner_mask_ids.unsqueeze(0)

        # Caption ids (+ optional CLIP)
        caption_ids = self.tokenizer(caption, return_tensors="pt", truncation=True, max_length=100)["input_ids"]
        if self.use_clip_tokenizer:
            import clip
            caption_ids_clip = clip.tokenize(caption, truncate=True)
        else:
            caption_ids_clip = None

        # Build entity ID tensors for article & caption (flatten and separate)
        names_art_ids, _ = make_new_entity_ids(article, names_art, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)
        org_norp_gpe_loc_art_ids, _ = make_new_entity_ids(article, org_norp_gpe_loc_art, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)

        names_ids_flatten, names_ids = make_new_entity_ids(caption, names_cap, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)
        org_norp_gpe_loc_ids_flatten, org_norp_gpe_loc_ids = make_new_entity_ids(caption, org_norp_gpe_loc_cap, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)

        # Person id spans
        person_id_positions = get_person_ids_position(article_ner_mask_dict["input_ids"], person_token_id=self.person_token_id, article_max_length=self.max_len)
        # For caption: rebuild caption with entity-type replacement and then mark spans
        cap_mask_ids = make_new_article_ids_all_ent(caption, [*names_cap, *org_cap, *gpe_loc_cap],
                                                    [*(["PERSON"]*len(names_cap)), *(["ORG"]*len(org_cap)), *(["LOC"]*len(gpe_loc_cap))],
                                                    self.tokenizer)["input_ids"]
        person_id_positions_cap = get_person_ids_position(cap_mask_ids, person_token_id=self.person_token_id, article_max_length=20, is_tgt_input=True)

        # Images
        img_tensor = self.transform(img).unsqueeze(0)

        # Features not present in demo.json => empty
        face_emb = np.array([[]])
        obj_emb = np.array([[]])

        return {
            "article": article,
            "article_ids": article_ids,
            "article_ner_mask_ids": article_ner_mask_ids,
            "caption": caption,
            "caption_ids": caption_ids,
            "caption_ids_clip": caption_ids_clip,
            "names_art": names_art,
            "org_norp_gpe_loc_art": org_norp_gpe_loc_art,
            "names_art_ids": names_art_ids,
            "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids,
            "names": names_cap,
            "org_norp_gpe_loc": org_norp_gpe_loc_cap,
            "names_ids": names_ids,
            "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids,
            "all_gt_ner": all_gt_ner,
            "all_gt_ner_ids": gt_ner_ids,
            "face_emb": face_emb,
            "obj_emb": obj_emb,
            "img_tensor": img_tensor,
            "names_ids_flatten": names_ids_flatten,
            "org_norp_gpe_loc_ids_flatten": org_norp_gpe_loc_ids_flatten,
            "person_id_positions": person_id_positions,
            "person_id_positions_cap": person_id_positions_cap,
        }

    def __len__(self):
        return len(self.data_dict)


# Backward-compatible alias (so import sites using GoodNews still work)
GoodNewsDictDatasetEntityTypeFixLenEntPos = ViWikiDictDatasetEntityTypeFixLenEntPos


# ------------------------------
# Optional helper: augment a viwiki dict with name_pos_cap field
# ------------------------------

def add_name_pos_list_to_dict_viwiki(
    data_dict: Dict[str, Any],
    tokenizer: BartTokenizer,
    vncore_dir: Optional[str] = None,
    vncore_jar: str = "VnCoreNLP-1.1.1.jar",
) -> Dict[str, Any]:
    vnner = _VNCoreNER(save_dir=vncore_dir or "./vncorenlp", jar_name=vncore_jar)
    new_dict = {}
    for key, value in data_dict.items():
        new_dict[key] = copy.deepcopy(value)
        caption = value.get("caption", "")
        ner = vnner.extract(caption)
        names_cap = ner.get("PER", [])
        org_cap = ner.get("ORG", [])
        gpe_cap = ner.get("LOC", [])
        cap_mask_ids = make_new_article_ids_all_ent(caption, [*names_cap, *org_cap, *gpe_cap],
                                                    [*(["PERSON"]*len(names_cap)), *(["ORG"]*len(org_cap)), *(["LOC"]*len(gpe_cap))],
                                                    tokenizer)["input_ids"]
        position_list = get_person_ids_position(cap_mask_ids, person_token_id=50265, article_max_length=20, is_tgt_input=True)
        new_dict[key]["name_pos_cap"] = position_list
    return new_dict


if __name__ == "__main__":
    # Minimal example that reads a demo.json-style file and writes *_cap_name_pos.json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="demo.json-like dict")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--tokenizer", default="/data2/npl/ICEK/vacnic/bartpho-syllable-base", local_files_only=True)
    ap.add_argument("--vncore_dir", default="./vncorenlp", help="Folder containing VnCoreNLP jar and models")
    ap.add_argument("--vncore_jar", default="VnCoreNLP-1.1.1.jar", help="VnCoreNLP jar filename inside --vncore_dir")

    args = ap.parse_args()

    tok = BartTokenizer.from_pretrained(args.tokenizer)
    tok.add_special_tokens({"additional_special_tokens":['<PERSON>', '<ORGNORP>', '<GPELOC>', '<NONAME>']})

    with open(args.input_json, "r", encoding="utf-8") as f:
        dd = json.load(f)

    out = add_name_pos_list_to_dict_viwiki(dd, tok, vncore_dir=args.vncore_dir, vncore_jar=args.vncore_jar)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
