from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import os
from io import BytesIO

import clip
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import faiss
from pathlib import Path

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 
app.static_folder = 'templates'

################################ clip_retrieval ##########################
# Replace this with the actual path to your images
IMAGE_FOLDER = '/home/faheem/Workspace/CVSSP_Retrieval/notebook/image_folder/00000'

# IMAGE_PATHS_FILE = '/home/faheem/Workspace/CVSSP_Retrieval/image_paths.txt'
# with open(IMAGE_PATHS_FILE, 'r') as file:
#     image_paths = file.read().splitlines()
# image_list = [{"url": "/images/"+path.split('/')[-1], "title": os.path.basename(path), "description": "CUTE CATS WHO DOESNT LIKE THEM"} for path in image_paths][:30]

img_ind = faiss.read_index("/home/faheem/Workspace/CVSSP_Retrieval/notebook/img.index")
text_ind = faiss.read_index("/home/faheem/Workspace/CVSSP_Retrieval/notebook/text.index")
combined_ind = faiss.read_index("/home/faheem/Workspace/CVSSP_Retrieval/notebook/combined.index")
data_dir = Path("/home/faheem/Workspace/CVSSP_Retrieval/notebook/embeddings")

df = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

image_list = df["image_path"].tolist()
caption_list = df["caption"].tolist()
url_list = df["url"].tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

########################### AdaCLIP retrieval #######################
from modeling.loss import CrossEn
from modeling.model import AdaCLIP
from modeling.clip_model import CLIP
from modeling.metrics import t2v_metrics, v2t_metrics
from datasets.dataset import BaseDataset
from datasets.prefetch import PrefetchLoader
from configs.config import parser, parse_with_config
from argparse import Namespace

from transformers import CLIPTokenizer
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ada_retrieval import setup_model, setup_dataloaders, validate, query_retrievalv2

cfg = json.load(open('/home/faheem/Workspace/CVSSP_Retrieval/configs/custom_msrvtt_cfg.json'))
cfg = Namespace(**cfg)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

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
    ret_metrics, _, frame_embd, text_embd, lengths = validate(ada_model, eval_loader, device, cfg, gflops_table=gflops_table)
    torch.save(frame_embd, file_path+'frame_embd.pt')
    torch.save(text_embd, file_path+'text_embd.pt')
    torch.save(lengths, file_path+'lengths.pt')
    with open(file_path+'ret_metrics.json', 'w') as f:
        json.dump(ret_metrics, f, indent=4)
    print(f"Saved tensor to {file_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text', methods=['POST'])
def handle_text():
    data = request.form
    text = data.get('text')

    # Process the text
    text_tokens = clip.tokenize([text], truncate=True)

    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype('float32')

    D, I = text_ind.search(text_embeddings, 5)
    
    output = []
    for d, i in zip(D[0], I[0]):
        output.append({
            "Similarity": round(float(d), 5),
            "Index": int(i),
            "Caption": caption_list[i],
            "Image url": url_list[i],
        })

    return jsonify({"texts": output})

@app.route('/image', methods=['POST'])
def handle_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files['image']

    # Process the image
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    image_path = os.path.join(upload_folder, 'temp.jpg')
    image.save(image_path)

    image = Image.open(image_path)
    image_tensor = preprocess(image)
    os.remove(image_path)

    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    image_features /= image_features.norm(dim=-1, keepdim=True)

    image_embeddings = image_features.cpu().detach().numpy().astype('float32')

    D, I = img_ind.search(image_embeddings, 5)

    output = []
    for d, i in zip(D[0], I[0]):
        output.append({
            "Similarity": round(float(d), 5),
            "Index": int(i),
            "Caption": caption_list[i],
            "Image url": url_list[i],
        })
        
    return jsonify({"images": output})

@app.route('/videotext/<path:filename>')
def serve_video(filename):
    test = send_from_directory('/home/faheem/Workspace/AdaCLIP/data/MSRVTT/videos/all/', filename)
    return test


@app.route('/both', methods=['POST'])
def handle_both():
    data = request.form
    text = data.get('text')
    emphasis = float(data.get('emphasis', 0.5))
    video_checked = data.get('video') == 'true'
    if 'image' in request.files and video_checked:
        return jsonify({"error": "Cannot upload both image and video"}), 400
    if 'image' in request.files:
        image = request.files['image']
        # Process the text and image
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        image_path = os.path.join(upload_folder, 'temp.jpg')
        image.save(image_path)

        image = Image.open(image_path)
        image_tensor = preprocess(image)

        image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_embeddings = image_features.cpu().detach().numpy().astype('float32')

        text_tokens = clip.tokenize([text], truncate=True)

        text_features = model.encode_text(text_tokens.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float32')

        combined_embeddings = (1-emphasis) * image_embeddings + emphasis * text_embeddings

        D, I = combined_ind.search(combined_embeddings, 5)
        output = []
        for d, i in zip(D[0], I[0]):
            output.append({
                "Similarity": round(float(d), 5),
                "Index": int(i),
                "Caption": caption_list[i],
                "Image url": url_list[i],
            })
        return jsonify({"images": output})

    if video_checked:
        print("Video option is checked")  # Dummy print command
        # Process the video
        results = query_retrievalv2(cfg, text, device, ada_model, frame_embd, lengths, top_k=5)
        # Process the text and video
        output = []
        print(f"query is: {text}")
        for i, (sim, caption, video_path) in enumerate(results):
            print(video_path)
            output.append({
                "Similarity": round(float(sim), 5),
                "Index": int(i),
                "Caption": caption,
                "Video_url": f"/videotext/{video_path.split('/')[-1]}",
            })
        return jsonify({"videos": [output]})
    
    return jsonify({"error": "No image uploaded or video selected"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)