
import argparse
from cmath import nan
import logging
import random
import torch
import torch.nn as nn
parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=str, default=684331)
parser.add_argument("--gpu_ids", type=str, default="1")
parser.add_argument("--num_workers", type=int, default=4)
# TEST DATASET
parser.add_argument("--train_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/demo20.json')
# parser.add_argument("--val_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/demo20.json')
# parser.add_argument("--test_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/demo20.json')
# NEW DATASET
# parser.add_argument("--train_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/train.json')
parser.add_argument("--val_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/val.json')
parser.add_argument("--test_json", type=str, default=r'/datastore/npl/ICEK/vacnic/data/test.json')

parser.add_argument("--article_max_length", type=int, default=512)
parser.add_argument("--caption_max_length", type=int, default=100)
parser.add_argument("--plm_type", type=str, default=r"/datastore/npl/ICEK/vacnic/vacnic_pretrained_model/bartpho-syllable")
parser.add_argument("--clip_type", type=str, default="ViT-B/32")
parser.add_argument("--ent_start_token", type=str, default="<ENT>")
parser.add_argument("--ent_end_token", type=str, default="<ENT>")
parser.add_argument("--perturb", type=bool, default=False)

parser.add_argument("--enc_fusion_layer",default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], type=int)
parser.add_argument("--dim_common", type=int, default=1024)

parser.add_argument("--warmup_rate", type=float, default=0.05)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--num_epoch", type=int, default=200) #######################

# ---- Early stopping (optional) ----
parser.add_argument("--early_stop", default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--early_stop_patience", type=int, default=10, help="Stop if val loss doesn't improve for N epochs")
parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum loss decrease to count as improvement")
parser.add_argument("--early_stop_warmup", type=int, default=0, help="Ignore early-stop checks for first N epochs")

parser.add_argument("--lr_bart", type=float, default = 3e-5)
parser.add_argument("--lr_clip", type=float, default = 1e-7)
parser.add_argument("--weight_decay", type=float, default = 0.01)
parser.add_argument("--clip_norm", type=float, default = 0.1)

parser.add_argument("--emb_dir", type=str, default= r"/datastore/npl/ICEK/vacnic/data/embedding")
parser.add_argument("--art_dir", type=str, default= r"/datastore/npl/ICEK/vacnic/data/embedding")
parser.add_argument("--img_dir", type=str, default= r"/datastore/npl/ICEK/Wikipedia/image_resized")
parser.add_argument("--out_dir", type=str, default= r"/datastore/npl/ICEK/vacnic/output/v2")

parser.add_argument("--mapping_loss_type", type=str, default= "contrastive")

parser.add_argument("--trained_clip", type=str, default="no")
parser.add_argument("--no_clip_loss", default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--prompt_size", type=int, default=20)
parser.add_argument("--use_vis_cls", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--max_ner_type_len", type=int, default=80)
parser.add_argument("--max_ner_type_len_gt", type=int, default=20)

parser.add_argument("--freeze_clip", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--prompt_mlp_type", type=str, default="clipcap")
parser.add_argument("--map_size", default= [196, 256, 64, 16], nargs="+", type=int)

parser.add_argument("--no_mapping", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--mapping_loss_weight", type=float, default = 1.0)
parser.add_argument("--img_size", type=int, default=768)

parser.add_argument("--only_image", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--use_secla", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--num_sentences", type=int, default=8)

parser.add_argument("--offline_wandb", default=True, type=lambda x: (str(x).lower() == 'true'))

# parser.add_argument("--perturb", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--no_clip_norm", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--init_attn_weight", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.5)

args = parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_model_parameters(model):
    print("Scanning model parameters for anomalies...")
    for name, param in model.named_parameters():
        # check for NaNs
        if torch.isnan(param).any():
            print(f"!! NaN detected in {name}")
            
        # check for uninitialized garbage (magnitude > 1000 is usually suspicious for weights)
        max_val = param.abs().max()
        if max_val > 1000:
            print(f"!! SUSPICIOUS MAGNITUDE in {name}: {max_val:.2e}")

def sanitize_model_weights(model, log_failures=True):
    """
    Scans the model for 'garbage' values (uninitialized memory) caused by 
    low_cpu_mem_usage or missing checkpoint keys, and resets them.
    """
    print(f"SCANNING MODEL ON DEVICE: {next(model.parameters()).device}...")
    
    has_fixed = False
    
    # We loop through every module (layer) in the network
    for name, module in model.named_modules():
        
        # Target your custom LayerNorms specifically
        if isinstance(module, nn.LayerNorm):
            # 1. Check Weight
            if hasattr(module, 'weight') and module.weight is not None:
                # Calculate average magnitude. If it's huge (>1000) or NaN, it's garbage.
                # standard LayerNorm weights should be close to 1.0.
                w_mean = module.weight.detach().abs().mean()
                is_garbage = w_mean > 100 or torch.isnan(w_mean)
                
                if is_garbage:
                    if log_failures:
                        print(f"!! FIXING GARBAGE WEIGHT in {name} (Mean: {w_mean:.2e})")
                    
                    # FORCE RESET TO IDENTITY
                    with torch.no_grad():
                        module.weight.fill_(1.0)
                    has_fixed = True

            # 2. Check Bias
            if hasattr(module, 'bias') and module.bias is not None:
                b_mean = module.bias.detach().abs().mean()
                is_garbage = b_mean > 100 or torch.isnan(b_mean)
                
                if is_garbage:
                    if log_failures:
                        print(f"!! FIXING GARBAGE BIAS in {name} (Mean: {b_mean:.2e})")
                    
                    # FORCE RESET TO ZERO
                    with torch.no_grad():
                        module.bias.fill_(0.0)
                    has_fixed = True

    if not has_fixed:
        print("Scan complete. No uninitialized layers found.")
    else:
        print("Scan complete. Garbage layers have been re-initialized.")

def prep_for_training(model, train_size, DEVICE):
    model.to(DEVICE)
    sanitize_model_weights(model)
    check_model_parameters(model)
    # clip_model.to(DEVICE)
    if "," in args.gpu_ids:
        optimizer_bart = optim.AdamW(list(model.module.model.parameters()) + list(model.module.lm_head.parameters()),betas= (0.9, 0.999), lr=args.lr_bart, eps=1e-8, weight_decay=args.weight_decay)

        optimizer_clip = optim.AdamW(list(model.module.clip_model.parameters()),betas= (0.9, 0.999), lr=args.lr_clip, eps=1e-8, weight_decay=args.weight_decay)
    else:
        optimizer_bart = optim.AdamW(list(model.model.parameters()) + list(model.lm_head.parameters()),betas= (0.9, 0.999), lr=args.lr_bart, eps=1e-8, weight_decay=args.weight_decay)

        optimizer_clip = optim.AdamW(list(model.clip_model.parameters()),betas= (0.9, 0.999), lr=args.lr_clip, eps=1e-8, weight_decay=args.weight_decay)

    num_training_steps = args.num_epoch * train_size / args.train_batch_size
    num_warmup_steps = args.warmup_rate * num_training_steps
    
    scheduler_bart = get_linear_schedule_with_warmup(optimizer_bart,
                                                num_warmup_steps,
                                                num_training_steps)
    scheduler_clip = get_linear_schedule_with_warmup(optimizer_clip,
                                                num_warmup_steps,
                                                num_training_steps)

    return model, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip

def get_embedding_ner(model, ner_ids_3d):
    bsz, num_ner, id_len = ner_ids_3d.size()
    hidden_states_ner_list = []
    with torch.no_grad():
        if "," in args.gpu_ids:
            encoder = model.module.model.encoder
        else:
            encoder = model.model.encoder
        for i in range(num_ner):
            ner_ids = ner_ids_3d[:, i, :].squeeze(1)
            ner_shape = ner_ids.size()
            ner_ids = ner_ids.view(-1, ner_shape[-1])
            ner_embeds = encoder.embed_tokens_ner(ner_ids) * encoder.embed_scale
            embed_pos_ner = encoder.embed_positions_ner(ner_shape)

            hidden_states_ner = ner_embeds + embed_pos_ner
            hidden_states_ner = encoder.layernorm_embedding_ner(hidden_states_ner)
            hidden_states_ner = torch.nn.functional.dropout(hidden_states_ner, p=encoder.dropout, training=False)
            hidden_states_ner_list.append(torch.mean(hidden_states_ner, dim=1))
            del hidden_states_ner
    # return hidden_states_ner
    return torch.stack(hidden_states_ner_list, dim=1)


def pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    emb = torch.nan_to_num(emb, nan=1.0)
    return emb


def shift_tokens_right(input_ids, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def create_src_mask_bart(input_ids):
    src_padding_mask = (input_ids == 1)
    src_mask = (src_padding_mask<1)
    src_mask = src_mask.int().type(torch.int64)
    src_mask = src_mask.to(input_ids.device)
    return src_mask

def extract_clip_img_feat(clip_model, x):
    """
    x: [B, 3, H, W], already preprocessed (processor(... )["pixel_values"])
    returns:
        x_tokens: [B, num_patches, D]
        x_cls:    [B, D]
    """
    with torch.no_grad():
        clip_model.eval()

        vision_model = clip_model.vision_model
        dtype = next(vision_model.parameters()).dtype
        x = x.to(dtype)

        outputs = vision_model(pixel_values=x)

        last_hidden = outputs.last_hidden_state       # [B, 1 + num_patches, D]
        x_cls = outputs.pooler_output                 # [B, D]
        x_tokens = last_hidden[:, 1:, :]              # [B, num_patches, D]

        return x_tokens.float(), x_cls.float(), last_hidden.float()


def train_epoch(bart_model, model, loss_margin, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart, train_dataloader, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip, epoch, DEVICE):
    model.train()
    tr_loss = 0
    tr_txt_loss = 0
    tr_clip_loss = 0
    tr_face_name_loss = 0
    tr_ner_loss = 0
    nb_tr_steps = 0

    tr_margin_loss = 0
    bi_contras_loss = BatchSoftmax()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        src_ids, tgt_ids, tgt_ids_clip, img_tensors, face_emb, names_art_ids, names_ids, names_ids_flatten = batch["article_ids"], batch["caption_ids"], batch["caption_ids_clip"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"], batch["names_ids"], batch["names_ids_flatten"]
        src_ids = src_ids.to(DEVICE)
        # src_ner_mask_ids = src_ner_mask_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)
        names_ids_3d = names_ids.to(DEVICE)
        names_ids_flatten = names_ids_flatten.to(DEVICE)

        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)
        names_cap_mask = create_src_mask_bart(names_ids_flatten)

        if "," in args.gpu_ids:
            img_feat, img_feat_cls, _ = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat, img_feat_cls, _ = extract_clip_img_feat(model.clip_model, img_tensors)

# # --- DEBUG BLOCK: Check Inputs Before Model Call ---
#         print(f"\n>>> Checking inputs for Batch {batch['article_ids'].shape}...")

#         def check_input_tensor(name, t):
#             if t is None:
#                 print(f" {name:<25} | IS NONE (Skipping)")
#                 return
            
#             if not isinstance(t, torch.Tensor):
#                 print(f" {name:<25} | Type: {type(t)} (Not a tensor)")
#                 return

#             # CRITICAL FIX: Cast to float for stats to avoid "mean() not implemented for Long"
#             t_float = t.float()

#             # Check for numerical instability
#             is_nan = torch.isnan(t_float).any().item()
#             is_inf = torch.isinf(t_float).any().item()
            
#             # Basic Stats
#             if t.numel() > 0:
#                 t_min = t_float.min().item()
#                 t_max = t_float.max().item()
#                 t_mean = t_float.mean().item()
#             else:
#                 t_min, t_max, t_mean = 0, 0, 0

#             status = "OK"
#             if is_nan: status = "!!! FAIL: NaN DETECTED!!!"
#             elif is_inf: status = "!!! FAIL: Inf DETECTED!!!"
            
#             # Only warn about explosion for floating point tensors (images/features), not IDs
#             elif t.is_floating_point():
#                 if abs(t_max) > 65000: status = "! CRITICAL: FP16 OVERFLOW RISK"
#                 elif abs(t_max) > 1e4: status = "! WARNING: High Value"

#             print(f" {name:<25} | Shape: {str(list(t.shape)):<18} | "
#                   f"Min: {t_min:<10.4f} | Max: {t_max:<10.4f} | Status: {status}")
            
#             if is_nan or is_inf:
#                 raise RuntimeError(f"Training Halted: {name} contains invalid values (NaN/Inf).")

#         # Check every tensor argument
#         check_input_tensor("input_ids", src_ids)
#         check_input_tensor("attention_mask", src_mask)
#         check_input_tensor("decoder_input_ids", tgt_input)
        
#         # CRITICAL CHECKS (Most likely culprits)
#         check_input_tensor("image_features", img_feat_cls) 
#         check_input_tensor("face_features", face_emb)
#         check_input_tensor("face_mask", face_mask)
        
#         check_input_tensor("name_ids", names_art_ids)
#         check_input_tensor("name_mask", names_art_mask)
        
#         print(">>> All inputs verified. Proceeding to forward pass...\n")
#         # --- END DEBUG BLOCK ---

        if args.prompt_mlp_type == "clipcap":
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat_cls, face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        else:
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat, face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        logits = output["logits"]

        # print("[DEBUG] logits has NaN:", torch.isnan(logits).any().item())
        # print("[DEBUG] logits has Inf:", torch.isinf(logits).any().item())
        # print("[DEBUG] logits min/max:", logits.min().item(), logits.max().item())
        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
        # print(f"[DEBUG] tgt_ids shape {tgt_ids.shape}, logits shape {logits.shape}")
        tr_txt_loss += txt_loss.item()

        # import ipdb
        # ipdb.set_trace()
        decoder_hidden_states = output["decoder_hidden_states"][-1]
        output_bart = bart_model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input)
        decoder_hidden_states_bart = output_bart["decoder_hidden_states"][-1]

        tgt_mask = create_src_mask_bart(tgt_ids)
        decoder_hidden_states = pool(decoder_hidden_states, tgt_mask)
        decoder_hidden_states_bart = pool(decoder_hidden_states_bart, tgt_mask)
        
        decoder_hidden_states = decoder_hidden_states / decoder_hidden_states.norm(dim=1, keepdim=True)
        decoder_hidden_states_bart = decoder_hidden_states_bart / decoder_hidden_states_bart.norm(dim=1, keepdim=True)


        # loss_bart_margin = loss_margin(decoder_hidden_states, decoder_hidden_states_bart, -torch.ones(decoder_hidden_states.shape[0]).to(decoder_hidden_states.device))
        scores = torch.matmul(decoder_hidden_states, decoder_hidden_states_bart.t())

        loss_bart_margin = loss_margin(scores.diag(), -torch.ones(decoder_hidden_states.shape[0]).to(decoder_hidden_states.device))

        tr_margin_loss += loss_bart_margin.item()
        # ipdb.set_trace()

        if not args.no_clip_loss:
            if "," in args.gpu_ids:
                logits_per_image, logits_per_text = model.module.clip_model(img_tensors, tgt_ids_clip)
            else:
                logits_per_image, logits_per_text = model.clip_model(img_tensors, tgt_ids_clip)
            clip_gt = torch.arange(img_tensors.size()[0], dtype=torch.long, device=DEVICE)

            total_loss_clip = (loss_img_clip(logits_per_image, clip_gt) + loss_txt_clip(logits_per_text, clip_gt))/2

            tr_clip_loss += total_loss_clip.item()
        

        if not args.no_mapping:
            if args.use_secla:
                hidden_states_face = output["hidden_states_face"]
                hidden_states_names = get_embedding_ner(model=model, ner_ids_3d=names_ids_3d)
                hidden_states_names = hidden_states_names.to(DEVICE)
                # print(f"[DEBUG] hidden_states_face {hidden_states_face.shape}, hidden_states_names {hidden_states_names.shape}")
                face_name_loss = bi_contras_loss(hidden_states_face, hidden_states_names)
                tr_face_name_loss += face_name_loss.item()

            else:
                hidden_states_face = output["hidden_states_face"]
                # face_feat_map = torch.mean(face_feat_map, dim=1)
                hidden_states_face = pool(hidden_states_face, face_mask)
                hidden_states_face = hidden_states_face / hidden_states_face.norm(dim=1, keepdim=True)

                with torch.no_grad():
                    hidden_states_names = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat_cls, face_features=face_emb, face_mask=face_mask, name_ids=names_ids_flatten, name_mask=names_cap_mask, add_ner_ffn=False)["hidden_states_ner"]
                
                hidden_states_names = hidden_states_names.to(DEVICE)
                # hidden_states_names = pool_replace(hidden_states_names, pooling_mask_ent_gt.to(DEVICE), hidden_states_face)
                hidden_states_names = pool(hidden_states_names, names_cap_mask.to(DEVICE))
                hidden_states_names = hidden_states_names / hidden_states_names.norm(dim=1, keepdim=True)

                if "," in args.gpu_ids:
                    logit_contras1 = model.module.clip_model.logit_scale.exp() * hidden_states_names @ hidden_states_face.t()
                    logit_contras2 = model.module.clip_model.logit_scale.exp() * hidden_states_face @ hidden_states_names.t()
                else:
                    logit_contras1 = model.clip_model.logit_scale.exp() * hidden_states_names @ hidden_states_face.t()
                    logit_contras2 = model.clip_model.logit_scale.exp() * hidden_states_face @ hidden_states_names.t()
                clip_gt = torch.arange(img_tensors.size()[0], dtype=torch.long, device=DEVICE)
                face_name_loss = 0.5 * loss_clip_bart(logit_contras1, clip_gt) + 0.5 * loss_clip_bart(logit_contras2, clip_gt)
                    
                tr_face_name_loss += face_name_loss.item()

        
        if not args.no_clip_loss:
            loss = total_loss_clip + txt_loss + args.mapping_loss_weight * face_name_loss + args.alpha * loss_bart_margin
        elif args.no_mapping:
            loss = txt_loss + args.alpha * loss_bart_margin
        else:
            loss = txt_loss + args.mapping_loss_weight * face_name_loss + args.alpha * loss_bart_margin
        # print(f"[DEBUG] loss {loss} \n txt_loss {txt_loss} \n args.mapping_loss_weight {args.mapping_loss_weight} \n face_name_loss {face_name_loss} \n args.alpha {args.alpha} \n loss_bart_margin {loss_bart_margin} ")
        loss.backward()
        if not args.no_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
        tr_loss += loss.item()
        nb_tr_steps += 1
        if loss == nan:
            print(batch)
        
        optimizer_bart.step()
        scheduler_bart.step()
        optimizer_bart.zero_grad()

        logging.info({"loss": loss})
        logging.info({"text loss": txt_loss})
        logging.info({"face name loss": face_name_loss})
        logging.info({"margin loss": loss_bart_margin})


    return tr_loss / nb_tr_steps


def eval_epoch(model, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart, val_dataloader, DEVICE):
    
    model.eval()
    # clip_model.eval()
    val_loss = 0
    # val_clip_loss = 0
    # val_contras_loss = 0
    nb_val_steps = 0
    out_dict = {}    
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []
    for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        out_dict[step] = {}
        
        src_ids, tgt_ids, tgt_sent, tgt_ids_clip, img_tensors, face_emb, names_art_ids, = batch["article_ids"], batch["caption_ids"], batch["caption"], batch["caption_ids_clip"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"],
        src_ids = src_ids.to(DEVICE)
        # src_ner_mask_ids = src_ner_mask_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)

        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)



        if "," in args.gpu_ids:
            img_feat,img_feat_cls,_ = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat,img_feat_cls,_ = extract_clip_img_feat(model.clip_model, img_tensors)


        src_mask = create_src_mask_bart(src_ids)


        if args.prompt_mlp_type == "clipcap":
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat_cls, face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        else:
            output = model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input, image_features=img_feat, face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        logits = output["logits"]

        out_dict[step]["logit_output"] = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.argmax(logits[i], dim=-1))) for i in range(logits.shape[0])]

        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))

        loss = txt_loss
        out_dict[step]["gt_cap"] = tgt_sent

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        else:
            loss = loss
        val_loss += loss.item()
        nb_val_steps += 1

        logging.info({"validation loss": loss})
    

        # GEN CAPTION: 
        beam_size = args.beam_size
        max_length = args.max_length
        if "," in args.gpu_ids:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        else:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
            else:
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        # print("gen type:", type(gen_cap_ids))
        # print("keys:", list(gen_cap_ids.keys())[:10])
        gen_cap = tokenizer.batch_decode(gen_cap_ids.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"[DEBUG] gen_cap: {gen_cap}")
        gen_unidecode = gen_cap
        gt_unidecode = tgt_sent[0]

        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    logging.info({"bleu": out_dict["bleu"], "other metrics": out_dict["other metrics"]})
    return val_loss / nb_val_steps, out_dict


def train(bart_model, model, loss_margin, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart,
          train_dataloader, val_dataloader, test_dataloader,
          optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip,
          model_name, DEVICE):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    best_ckpt_path = os.path.join(args.out_dir, model_name + ".pt")

    # wandb.watch(model)
    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_epoch(
            bart_model, model, loss_margin, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart,
            train_dataloader, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip,
            epoch_i, DEVICE
        )

        val_loss, out_dict = eval_epoch(
            model, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart,
            val_dataloader, DEVICE
        )

        logging.info({"epoch": epoch_i, "train_loss": train_loss, "val_loss": val_loss})

        # ---- Check improvement ----
        improved = val_loss < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0

            model.eval()
            torch.save(model, best_ckpt_path)
            with open(os.path.join(args.out_dir, f"{epoch_i}{model_name}v.json"), "w", encoding="utf-8") as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=4)
            logging.info({"best_val_loss": best_val_loss, "best_epoch": epoch_i})
        else:
            # only count "no improve" after warmup, and if early stopping is enabled
            if args.early_stop and epoch_i >= args.early_stop_warmup:
                epochs_no_improve += 1
                logging.info({"epochs_no_improve": epochs_no_improve})

        # Always save last checkpoint
        torch.save(model, os.path.join(args.out_dir, model_name + "last.pt"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---- Early stop ----
        if args.early_stop and epoch_i >= args.early_stop_warmup and epochs_no_improve >= args.early_stop_patience:
            logging.info({
                "early_stop": True,
                "stopped_epoch": epoch_i,
                "best_val_loss": best_val_loss,
                "best_ckpt": best_ckpt_path
            })
            print(f"[EARLY STOP] epoch={epoch_i} best_val_loss={best_val_loss:.6f} ckpt={best_ckpt_path}")
            break

    return train_losses, val_losses



def gen_caption_from_loader_bart(model, data_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    out_dict = {}
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}

        src_ids, tgt_sent, img_tensors, face_emb, names_art_ids, = batch["article_ids"], batch["caption"], batch["img_tensor"], batch["face_emb"], batch["names_art_ids"],
        src_ids = src_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)

        names_art_ids = names_art_ids.to(DEVICE)

        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)


        if "," in args.gpu_ids:
            img_feat,img_feat_cls,_ = extract_clip_img_feat(model.module.clip_model, img_tensors)
        else:
            img_feat,img_feat_cls,_ = extract_clip_img_feat(model.clip_model, img_tensors)

        src_mask = create_src_mask_bart(src_ids)
        
        ner_mask = torch.ones((args.test_batch_size, args.max_ner_type_len_gt))
        ner_mask = ner_mask.to(DEVICE)

        if "," in args.gpu_ids:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
            else:
                gen_cap_ids = model.module.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
        else:
            if args.prompt_mlp_type == "clipcap":
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat_cls,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)
            else:
                gen_cap_ids = model.generate(input_ids=src_ids, attention_mask=src_mask, num_beams=beam_size, max_length=max_length, image_features=img_feat,  face_features=face_emb, face_mask=face_mask, name_ids=names_art_ids, name_mask=names_art_mask, add_ner_ffn=True)

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # gt_unidecode = unidecode.unidecode(tgt_sent[0])
        # gen_unidecode = unidecode.unidecode(gen_cap)
        gen_unidecode = gen_cap
        gt_unidecode = tgt_sent[0]

        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode
    
    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()


    out_dict["bleu"] = {}
    out_dict["bleu"] = {"bleu1":blue_score[0],"bleu2":blue_score[1],"bleu3":blue_score[2],"bleu4":blue_score[3]}
    out_dict["other metrics"] = {}
    out_dict["other metrics"] = {"rouge":rouge_score, "meteor":meteor_score, "cider":cider_score}
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, meteor_score, cider_score



def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


if __name__ == "__main__":

    import torch.optim as optim
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    import os, json, types, re, torch
    from transformers import CLIPModel, CLIPProcessor
    import numpy as np
    from functools import partial
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from transformers import (
            AutoTokenizer, PreTrainedTokenizerFast,
            BartForConditionalGeneration, get_linear_schedule_with_warmup)
    from ViWiki_dataset import (
            ViWikiDictDatasetEntityTypeFixLenEntPos,
            collate_fn_viwiki_entity_type)
    from model import BartForMultiModalGeneration
    # --------------------------------------
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(True)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    logging.basicConfig(filename="training_v5.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    def batch_softmax(phrase_region_match):
        # phrase_region_match [B, B, span_len, R]: span_len: names, R: faces
        batch_size, _, num_spans, _ = phrase_region_match.size()

        # [B, B, span_len]
        phrase_region_max = phrase_region_match.max(-1).values

        # Logits [B, B]
        phrase_region_scores = phrase_region_max.sum(-1)
        # Normalize scores
        scale = torch.tensor(num_spans).expand(batch_size).unsqueeze(1).expand((batch_size, batch_size))
        scale = scale.to(phrase_region_scores.device)
        logits = phrase_region_scores.div(scale)

        targets = torch.arange(batch_size).to(logits.device)

        return torch.nn.functional.cross_entropy(logits, targets)
        

    class BatchSoftmax(torch.nn.Module):
        def __init__(self):
            super(BatchSoftmax, self).__init__()

        def forward(self, face_j, ner_j):
            # print(f"[DEBUG] ner_j {ner_j.shape}")
            # print(f"[DEBUG] face_j {face_j.shape}")
            face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
            ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
            loss1 = batch_softmax(face_ner_match)
            loss2 = batch_softmax(ner_face_match)
            loss = loss1 + loss2
            return loss
    
    
    if args.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if args.plm_type.startswith("ainize"):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_type)
    if args.trained_clip == "no":
        clip_path = r"/datastore/npl/ICEK/vacnic/vacnic_pretrained_model/clip-ViT-B-32/0_CLIPModel"
        clip_model = CLIPModel.from_pretrained(
                clip_path,
                local_files_only=True
            ).to("cuda")
        clip_preprocess = CLIPProcessor.from_pretrained(
                clip_path,
                local_files_only=True
            )

    print(f"[DEBUG] DONE LOAD PRETRAINED MODEL")
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    model = BartForMultiModalGeneration.from_pretrained(args.plm_type, output_hidden_states=True, enc_fusion_layer=args.enc_fusion_layer, dim_common=args.dim_common, img_size=args.img_size, prompt_mlp_type=args.prompt_mlp_type, map_size=args.map_size, prompt_size=args.prompt_size, clip_model=clip_model, freeze_clip=args.freeze_clip, max_ner_type_len=args.max_ner_type_len, max_ner_type_len_gt=args.max_ner_type_len_gt, only_image=args.only_image, init_attn_weight=args.init_attn_weight)
    bart_model = BartForConditionalGeneration.from_pretrained(args.plm_type, output_hidden_states=True)
    bart_model = bart_model.to(DEVICE)
    sanitize_model_weights(model)
    for param in bart_model.parameters():
        param.requires_grad = False
    for param in bart_model.parameters():
        if param.requires_grad:
            print(param)

    tokenizer.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>"]})
    # noname_id = tokenizer.convert_tokens_to_ids("<NONAME>")  
    model.resize_token_embeddings(len(tokenizer))

    # # 1. Check the config (less reliable)
    # print(f"Model config vocab size: {model.config.vocab_size}")

    # # 2. Check the actual embedding layer (most reliable)
    # actual_model_vocab_size = model.get_input_embeddings().num_embeddings
    # print(f"Model's ACTUAL embedding size: {actual_model_vocab_size}")

    # # You can also check the shape directly
    # shape_0 = model.get_input_embeddings().weight.shape[0]
    # print(f"Model embedding layer shape[0]: {shape_0}")
    # # This assertion should pass if everything is correct
    # assert len(tokenizer) == actual_model_vocab_size

    if args.perturb:
        bos_noise = torch.randn(1024)
        model.model.shared.weight.data[0] = model.model.shared.weight.data[0] + bos_noise

    del clip_model
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    # print(len(tokenizer))  # Tokenizer vocab size
    # print(model.config.vocab_size) 
    print(f"[DEBUG] DONE LOAD BART MODEL")
    

    tokenizer_dataset = AutoTokenizer.from_pretrained(args.plm_type)
    tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
    
    person_token_id = tokenizer_dataset.convert_tokens_to_ids('<PERSON>')
    print(f"[DEBUG] PERSON ids: {person_token_id}")
    def build_dataset(json_path,split):
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        return ViWikiDictDatasetEntityTypeFixLenEntPos(
                    data_dict,
                    args.emb_dir,
                    args.img_dir,
                    args.art_dir,
                    tokenizer_dataset,               
                    use_clip_tokenizer=True,
                    transform=img_transform,
                    max_article_len=args.article_max_length,
                    max_ner_type_len=args.max_ner_type_len,
                    max_ner_type_len_gt=args.max_ner_type_len_gt,
                    retrieved_sent=True,
                    person_token_id=person_token_id,
                    split=split,
                    clip_processor = clip_preprocess,
                    entity_token_start=args.ent_start_token, 
                    entity_token_end=args.ent_end_token
                    )
    # train_data = build_dataset(args.train_json, "train")
    val_data   = build_dataset(args.val_json, "val")
    test_data  = build_dataset(args.test_json, "test")  
    train_data = build_dataset(args.train_json, "demo20")
    # val_data   = build_dataset(args.val_json, "demo20")
    # test_data  = build_dataset(args.test_json, "demo20")  
    train_loader = DataLoader(train_data, args.train_batch_size,
                        num_workers=args.num_workers, collate_fn=collate_fn_viwiki_entity_type)

    val_loader   = DataLoader(val_data,   args.val_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn_viwiki_entity_type)

    test_loader  = DataLoader(test_data,  args.test_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn_viwiki_entity_type)
    print(f"[DEBUG] train size: {len(train_data)}val size: {len(val_data)}, test size = {len(test_data)}")
    # logging.info({"train size":len(train_data), "val size": len(val_data), "test size": len(test_data)})

    model, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip = prep_for_training(model, len(train_data), DEVICE)
    print(f"[DEBUG] DONE PREP FOR TRAINING")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(DEVICE)
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    loss_clip_bart = torch.nn.CrossEntropyLoss()
    loss_margin = torch.nn.HingeEmbeddingLoss(margin=args.margin)
    print(f"[DEBUG] DONE LOSS LOADING, START TRAINING")

    train(bart_model, model, loss_margin, loss_fn, loss_img, loss_txt, loss_clip_bart, train_loader, val_loader, test_loader, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip, 'first', DEVICE)

    # Reload best checkpoint for final test-time generation (so you don't evaluate the "last" epoch).
    best_ckpt = os.path.join(args.out_dir, "first.pt")
    if os.path.exists(best_ckpt):
        model = torch.load(best_ckpt, map_location=DEVICE)
        model = model.to(DEVICE)
        print(f"[DEBUG] Loaded best checkpoint: {best_ckpt}")

    
    
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
    meteor_scores = []

    test_out_dict, blue1, blue2, blue3, blue4, rouge_score, meteor_score, cider_score = gen_caption_from_loader_bart(model, test_loader, tokenizer, bleu_scorer, rouge_scorer, cider_scorer, meteor_scorer, args.beam_size, args.max_length, DEVICE)
    with open(os.path.join(args.out_dir, 'first'+"last.json"), "w", encoding="utf-8") as f:
        json.dump(test_out_dict, f, ensure_ascii=False, indent=4)
    
    logging.info({"bleu1":blue1, "bleu2":blue2, "bleu3":blue3, "bleu4":blue4, "rouge":rouge_score, "meteor":meteor_score, "cider":cider_score})

    tokenizer.save_pretrained(args.out_dir)
