# VACNIC-VN (Vietnamese VACNIC)

Vietnamese re-implementation of **VACNIC — Visually-Aware Context Modeling for News Image Captioning** adapted to a Vietnamese news/wiki setting. This repo replaces the English text stack with **PhoBERT (syllable)** for encoding and **VNCoreNLP** for NER, and targets the **OK-ViWiki-Refined** dataset.

- Paper (NAACL 2024): [Visually-Aware Context Modeling for News Image Captioning](https://arxiv.org/abs/2308.08325)  
- Upstream reference code: [tingyu215/VACNIC](https://github.com/tingyu215/VACNIC)  
- Vietnamese dataset: [NezuMiii/OK-ViWiki-Refined](https://huggingface.co/datasets/NezuMiii/OK-ViWiki-Refined)  
- Vietnamese NER: [VNCoreNLP](https://github.com/vncorenlp/VnCoreNLP)  
- Vietnamese encoder: [vinai/bartpho-syllable](https://huggingface.co/vinai/bartpho-syllable) (syllable)

> **Key differences vs. upstream**
>
> - English BART/MBART → **BARTPho (syllable)** for Vietnamese text encoding  
> - English NER → **VNCoreNLP** for Vietnamese NER  
> - English datasets (GoodNews/NYTimes) → **OK-ViWiki-Refined** (vi)
