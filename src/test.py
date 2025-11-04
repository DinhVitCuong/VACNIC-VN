from transformers import CLIPModel, CLIPProcessor
clip_path = r"Z:\DATN\model\vacnic_model\clip-ViT-B-32"
clip_model = CLIPModel.from_pretrained(
        clip_path,
        local_files_only=True
    ).to("cuda")
clip_preprocess = CLIPProcessor.from_pretrained(
        clip_path,
        local_files_only=True
    )