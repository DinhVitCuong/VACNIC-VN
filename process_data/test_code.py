import sys, transformers, torch
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_info()

print("Python:", sys.version)
print("Transformers:", transformers.__version__)

tok_fast = AutoTokenizer.from_pretrained(r"Z:\DATN\model\vacnic_model\bartpho-syllable", use_fast=True)
print("FAST tokens:", tok_fast.tokenize("Xin chào Việt Nam!"))
