from flask import Flask, request, jsonify, render_template, send_from_directory
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
app.static_folder = 'templates'

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
    # if (data.texts) {
    #     data.texts.forEach(text => {
    #         const textElement = document.createElement('p');
    #         textElement.textContent = text.Caption;
    #         resultsContainer.appendChild(textElement);
    #     });
    # }
    output = []
    for d, i in zip(D[0], I[0]):
        output.append({
            "Similarity": float(d),
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

    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    image_features /= image_features.norm(dim=-1, keepdim=True)

    image_embeddings = image_features.cpu().detach().numpy().astype('float32')

    D, I = img_ind.search(image_embeddings, 5)

    output = []
    for d, i in zip(D[0], I[0]):
        output.append({
            "Similarity": float(d),
            "Index": int(i),
            "Caption": caption_list[i],
            "Image url": url_list[i],
        })
        
    return jsonify({"images": output})

@app.route('/video', methods=['POST'])
def handle_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    video = request.files['video']
    # Process the video
    return jsonify({"videos": [video.filename]})

@app.route('/both', methods=['POST'])
def handle_both():
    data = request.form
    text = data.get('text')
    emphasis = float(data.get('emphasis', 0.5))
    if 'image' in request.files and 'video' in request.files:
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
                "Similarity": float(d),
                "Index": int(i),
                "Caption": caption_list[i],
                "Image url": url_list[i],
            })
        return jsonify({"images": output})
    if 'video' in request.files:
        video = request.files['video']
        # Process the text and video
        return jsonify({"texts": [text], "videos": [video.filename]})
    return jsonify({"error": "No image or video uploaded"}), 400

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)