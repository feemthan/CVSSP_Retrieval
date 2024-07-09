from flask import Flask, request, jsonify, render_template, send_from_directory
import os

app = Flask(__name__)
app.static_folder = 'templates'

# Replace this with the actual path to your images
IMAGE_FOLDER = '/home/feem/Workspace/cats/00000'
IMAGE_PATHS_FILE = '/home/feem/Workspace/image_paths.txt'

with open(IMAGE_PATHS_FILE, 'r') as file:
    image_paths = file.read().splitlines()
image_list = [{"url": "/images/"+path.split('/')[-1], "title": os.path.basename(path), "description": "CUTE CATS WHO DOESNT LIKE THEM"} for path in image_paths][:30]
# print(image_list[0])
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text', methods=['POST'])
def handle_text():
    data = request.form
    text = data.get('text')
    # Process the text
    return jsonify({"texts": [text]})

@app.route('/image', methods=['POST'])
def handle_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files['image']
    # Process the image
    return jsonify({"images":image_list})

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
    if 'image' in request.files and 'video' in request.files:
        return jsonify({"error": "Cannot upload both image and video"}), 400
    if 'image' in request.files:
        image = request.files['image']
        # Process the text and image
        return jsonify({"images":image_list})
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