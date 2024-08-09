import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.logger import LOGGER

from modeling.loss import CrossEn
from modeling.model import AdaCLIP
from modeling.clip_model import CLIP
from modeling.metrics import t2v_metrics, v2t_metrics
from datasets.dataset import BaseDataset
from datasets.prefetch import PrefetchLoader
from configs.config import parser, parse_with_config
from argparse import Namespace

import faiss
from transformers import CLIPTokenizer
import json
from types import SimpleNamespace

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def setup_model(cfg, device):
    LOGGER.info("Setup model...")

    pretrained_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    state_dict = {}
    epoch = 0
    if cfg.resume:
        LOGGER.info(f"Loading model checkpoint: {cfg.resume}...")
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        state_dict = checkpoint['state_dict']
        epoch = checkpoint["epoch"]
    else:
        LOGGER.info(f"Using CLIP pretrained weights...")
        for key, val in pretrained_state_dict.items():    
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        if cfg.sim_header != "meanP":
            for key, val in pretrained_state_dict.items():
                if key == "positional_embedding":
                    state_dict["frame_position_embeddings.weight"] = val.clone()

                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])

                    if num_layer < 4:
                        state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue

                    if num_layer == 4:
                        state_dict[key.replace(str(num_layer), "0")] = val.clone()

    model = AdaCLIP(cfg, pretrained_state_dict)
    model.load_state_dict(state_dict, strict=False)

    if cfg.freeze_clip:
        model.freeze_clip()
    if cfg.freeze_cnn and cfg.use_policy:
        model.sampler.freeze_cnn_backbone()

    model.to(device)

    LOGGER.info("Setup model done!")
    return model, epoch

def setup_dataloaders(cfg, device, train_annot, val_annot):
    LOGGER.info("Init. train_loader and val_loader...")

    train_dataset = BaseDataset(cfg, train_annot, is_train=True)
    val_dataset = BaseDataset(cfg, val_annot, is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=train_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=val_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    if str(device) != "cpu":
        train_loader = PrefetchLoader(train_loader)
        val_loader = PrefetchLoader(val_loader)
    
    LOGGER.info("Init. train_loader and val_loader done!")

    return train_loader, val_loader

def get_embeddings(val_loader, model, cfg):
    with torch.no_grad():
        text_embd = []
        frame_embd = []
        word_embd = []
        actions = []
        lengths = []
        break_pts = [0]
        # if is_main_process():
        pbar = tqdm(total=len(val_loader), desc="Evaluation", unit="batch")
        # else:
        #     pbar = NoOp()
        for minibatch in val_loader:
            output = model(minibatch["text_input_ids"], minibatch["clip_inputs"], minibatch["policy_inputs"], return_embds=True)
            text_embd.append(output["text_embd"])
            frame_embd.append(output["frame_embd"])
            word_embd.append(output["word_embd"])
            actions.append(output["actions"])
            lengths.append(output["lengths"])
            pbar.update(1)
        pbar.close()

        text_embd = torch.cat(text_embd, 0)
        frame_embd = torch.cat(frame_embd, 0)
        word_embd = torch.cat(word_embd, 0) if word_embd[0] is not None else None
        actions = torch.cat(actions, 0)
        lengths = torch.cat(lengths, 0)

        if break_pts == [0]:
            break_pts = None

        res = {
            "text_embd": text_embd,
            "frame_embd": frame_embd,
            "word_embd": word_embd,
            "actions": actions,
            "lengths": lengths,
        }

        return res, break_pts

def reshape_sim_matrix(sims, break_pts):
    num_t, num_v = sims.shape
    if num_t == num_v:
        return sims
    sims_reshaped = torch.zeros((num_v, num_v)).to(sims.device)
    for v in range(num_v):
        for i in range(len(break_pts)-1):
            sims_reshaped[i, v] = torch.max(sims[break_pts[i]:break_pts[i+1], v], dim=0)[0]
    return sims_reshaped

def compute_batched_sim_matrix(batch_size, model, text_embd, frame_embd, word_embd, lengths, runtime=False):
    sim_matrix = []
    text_batch_size = 1 if runtime else batch_size
    video_batch_size = frame_embd.shape[0] if runtime else batch_size
    with torch.no_grad():
        for ti in range(0, text_embd.shape[0], text_batch_size):
            tf = ti + text_batch_size
            text_embd_batch = text_embd[ti:tf]
            word_embd_batch = word_embd[ti:tf] if word_embd is not None else None
            lengths_batch = lengths[ti:tf]
            each_row = []
            for vi in range(0, frame_embd.shape[0], video_batch_size):
                vf = vi + video_batch_size
                frame_embd_batch = frame_embd[vi:vf]
                sims = model.compute_sim_matrix(frame_embd_batch, text_embd_batch, word_embd_batch, lengths_batch)
                each_row.append(sims)
            each_row = torch.concat(each_row, dim=-1)
            sim_matrix.append(each_row)
        sim_matrix = torch.concat(sim_matrix, dim=0)
    return sim_matrix

def compute_batched_query_sim_matrix(batch_size, model, query_embd, frame_embd, word_embd, lengths, runtime=False):
    sim_matrix = []
    text_batch_size = 1 if runtime else batch_size
    video_batch_size = frame_embd.shape[0] if runtime else batch_size
    with torch.no_grad():
        for ti in range(0, query_embd.shape[0], text_batch_size):
            tf = ti + text_batch_size
            query_embd_batch = query_embd[ti:tf]
            word_embd_batch = word_embd[ti:tf] if word_embd is not None else None
            lengths_batch = lengths[ti:tf]
            each_row = []
            for vi in range(0, frame_embd.shape[0], video_batch_size):
                vf = vi + video_batch_size
                frame_embd_batch = frame_embd[vi:vf]
                sims = model.compute_sim_matrix(frame_embd_batch, query_embd_batch, word_embd_batch, lengths_batch)
                each_row.append(sims)
            each_row = torch.concat(each_row, dim=-1)
            sim_matrix.append(each_row)
        sim_matrix = torch.concat(sim_matrix, dim=0)
    return sim_matrix

def compute_faiss_retrieval(batch_size, text_embd, frame_embd, top_k=5):
    # Convert embeddings to numpy and ensure they're float32
    frame_embd_np = frame_embd.cpu().numpy().astype(np.float32)
    text_embd_np = text_embd.cpu().numpy().astype(np.float32)

    frame_embd_np = np.mean(frame_embd_np, axis=1)
    text_embd_np = text_embd_np.squeeze(1)

    # Normalize embeddings (assuming cosine similarity is desired)
    faiss.normalize_L2(frame_embd_np)
    faiss.normalize_L2(text_embd_np)

    # Build FAISS index
    dimension = frame_embd_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(frame_embd_np)

    # Perform batch retrieval
    results = []
    for i in range(0, text_embd_np.shape[0], batch_size):
        batch = text_embd_np[i:i+batch_size]
        distances, indices = index.search(batch, top_k)
        results.append((distances, indices))

    # Combine results
    all_distances = np.concatenate([r[0] for r in results])
    all_indices = np.concatenate([r[1] for r in results])

    return all_distances, all_indices

@torch.no_grad()
def validate(model, val_loader, device, cfg, criterion=None, writer=None, epoch=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    if cfg.use_policy and cfg.warmup_epochs:
        model.sampler.top_k = cfg.top_k

    embds, break_pts = get_embeddings(val_loader, model, cfg)

    text_embd = embds["text_embd"]
    frame_embd = embds["frame_embd"]
    word_embd = embds["word_embd"]
    actions = embds["actions"]
    lengths = embds["lengths"]
    sims = compute_batched_sim_matrix(cfg.val_batch_size, model, text_embd, frame_embd, word_embd, lengths)
    LOGGER.info(f"Num. of queries: {sims.shape[0]}, Num. of videos: {sims.shape[1]}")

    tv_metrics = t2v_metrics(sims, break_pts)
    vt_metrics = v2t_metrics(sims, break_pts)
    all_metrics = {"t2v_metrics": tv_metrics, "v2t_metrics": vt_metrics}

    if criterion:
        reshaped_sims = reshape_sim_matrix(sims, break_pts)
        loss1 = criterion(reshaped_sims)
        loss2 = criterion(reshaped_sims.T)
        retrieval_loss = (loss1 + loss2) / 2
        writer.add_scalar('Retrieval Loss/val', retrieval_loss.item(), epoch)
        loss = retrieval_loss
        writer.add_scalar('Total Epoch Loss/val', loss.item(), epoch)
        LOGGER.info(f"EVAL epoch {epoch} Loss: {(loss.item()):.6f}")
        LOGGER.info(f"Retrieval Loss: {retrieval_loss.item():.3f}")
    actions = actions.cpu().detach().numpy()
    # log_policy_usage(actions, gflops_table, cfg, True)

    return all_metrics, actions, frame_embd, text_embd, lengths

def query_retrieval(cfg, query, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ada_model, epoch = setup_model(cfg, device=device)

    _, eval_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.test_annot) 

    file_path = 'embeddings/'

    if os.path.exists(file_path+"frame_embd.pt") and os.path.exists(file_path+"text_embd.pt") and os.path.exists(file_path+"lengths.pt") and os.path.exists(file_path+"ret_metrics.json"):
        frame_embd = torch.load(file_path+"frame_embd.pt", weights_only=True)
        text_embd = torch.load(file_path+'text_embd.pt', weights_only=True)
        lengths = torch.load(file_path+'lengths.pt', weights_only=True)
        with open(file_path+'ret_metrics.json', 'r') as f:
            ret_metrics = json.load(f)
        print(f"Loaded tensor from {file_path}")
    else:
        ret_metrics, _, frame_embd, text_embd, lengths = validate(ada_model, eval_loader, device, cfg)
        torch.save(frame_embd, file_path+'frame_embd.pt')
        torch.save(text_embd, file_path+'text_embd.pt')
        torch.save(lengths, file_path+'lengths.pt')
        with open(file_path+'ret_metrics.json', 'w') as f:
            json.dump(ret_metrics, f, indent=4)
        print(f"Saved tensor to {file_path}")

    LOGGER.info('Working on retrieval!')

    video_data_path = '/home/faheem/Workspace/AdaCLIP/data/MSRVTT/videos/all/'
    top_k = top_k
    query = query

    tokens = tokenizer(query, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokens.to(device)
    query_input_ids = tokens.input_ids.unsqueeze(0)

    with torch.no_grad():
        query_embd, word_embd = ada_model.get_text_output(query_input_ids, return_hidden=True)

    sims = compute_batched_query_sim_matrix(cfg.val_batch_size, ada_model, query_embd, frame_embd, word_embd, lengths)
    json_data = json.load(open(cfg.val_annot))

    distances, indices = compute_faiss_retrieval(cfg.val_batch_size, query_embd, frame_embd, top_k)
    video_data = [(video_info['sentences'][0], video_num) for video_num, video_info in json_data.items()]

    results = []
    # LOGGER.info(f"Top {top_k} video matches for the text query '{query}':")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        # LOGGER.info(f"  Rank {i+1}: Video frame {idx}, similarity score: {dist:.2f} Caption: {video_data[idx][0]} Path: {video_data_path}{video_data[idx][1]}.mp4")
        results.append([video_data[idx][0], video_data_path+video_data[idx][1]+".mp4"])
    LOGGER.info(results)
    return results

def query_retrievalv2(cfg, query, device, ada_model, frame_embd, lengths, top_k=5):

    top_k = top_k
    query = query
    video_data_path = '/home/faheem/Workspace/AdaCLIP/data/MSRVTT/videos/all/'

    tokens = tokenizer(query, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokens.to(device)
    query_input_ids = tokens.input_ids.unsqueeze(0)

    with torch.no_grad():
        query_embd, word_embd = ada_model.get_text_output(query_input_ids, return_hidden=True)

    sims = compute_batched_query_sim_matrix(cfg.val_batch_size, ada_model, query_embd, frame_embd, word_embd, lengths)
    json_data = json.load(open(cfg.val_annot))

    distances, indices = compute_faiss_retrieval(cfg.val_batch_size, query_embd, frame_embd, top_k)
    video_data = [(video_info['sentences'][0], video_num) for video_num, video_info in json_data.items()]

    results = []
    # LOGGER.info(f"Top {top_k} video matches for the text query '{query}':")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        # LOGGER.info(f"  Rank {i+1}: Video frame {idx}, similarity score: {dist:.2f} Caption: {video_data[idx][0]} Path: {video_data_path}{video_data[idx][1]}.mp4")
        results.append([dist, video_data[idx][0], video_data_path+video_data[idx][1]+".mp4"])
    LOGGER.info(results)
    return results

if __name__ == '__main__':
    cfg = json.load(open('/home/faheem/Workspace/CVSSP_Retrieval/configs/custom_msrvtt_cfg.json'))
    cfg = Namespace(**cfg)

    file_path = 'embeddings/'

    if os.path.exists(file_path+"frame_embd.pt") and os.path.exists(file_path+"text_embd.pt") and os.path.exists(file_path+"lengths.pt") and os.path.exists(file_path+"ret_metrics.json"):
        frame_embd = torch.load(file_path+"frame_embd.pt", weights_only=True)
        text_embd = torch.load(file_path+'text_embd.pt', weights_only=True)
        lengths = torch.load(file_path+'lengths.pt', weights_only=True)
        with open(file_path+'ret_metrics.json', 'r') as f:
            ret_metrics = json.load(f)
        print(f"Loaded tensor from {file_path}")
    else:
        ret_metrics, _, frame_embd, text_embd, lengths = validate(ada_model, eval_loader, device, cfg)
        torch.save(frame_embd, file_path+'frame_embd.pt')
        torch.save(text_embd, file_path+'text_embd.pt')
        torch.save(lengths, file_path+'lengths.pt')
        with open(file_path+'ret_metrics.json', 'w') as f:
            json.dump(ret_metrics, f, indent=4)
        print(f"Saved tensor to {file_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ada_model, epoch = setup_model(cfg, device=device)
    query_retrieval(cfg, query='cat drinking water', top_k=5)
    query_retrievalv2(cfg, query='cat drinking water', device=device, ada_model=ada_model, frame_embd=frame_embd, lengths=lengths, top_k=5)

