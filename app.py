import os
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import threading

app = Flask(__name__)
CORS(app)

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')

print("Loading CLIP Model (For High Accuracy Grid Classification)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

hcaptcha_pending = {}
hcaptcha_trained = {}

if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            hcaptcha_trained = data.get('trained', {})
    except Exception as e:
        print("DB Load Error:", e)

def persist_database():
    try:
        with open(DB_FILE, 'w') as f:
            json.dump({'pending': hcaptcha_pending, 'trained': hcaptcha_trained}, f)
    except:
        pass

def base64_to_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data)).convert("RGB")

def evaluate_auto_solve(task):
    if task['taskId'] in hcaptcha_trained:
        return {'solved': True, 'clicks': hcaptcha_trained[task['taskId']].get('clicks', [])}

    prompt_text = task.get('prompt', '').split('|||')[0].strip()
    media_list = task.get('media', [])
    if not media_list or len(media_list) <= 1:
        return {'solved': False} # Only processing grid tasks safely with CLIP

    images = []
    indices = []
    for m in media_list:
        if m.get('src'):
            try:
                img = base64_to_image(m['src'])
                images.append(img)
                indices.append(m['index'])
            except:
                pass
    
    predicted_clicks = []
    if images:
        try:
            inputs = clip_processor(text=[prompt_text], images=images, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            
            for i, score in enumerate(logits_per_image):
                if score[0].item() > 23.5:
                    predicted_clicks.append(indices[i])
        except Exception as e:
            print("CLIP Error:", e)

    if 1 <= len(predicted_clicks) <= 6:
        hcaptcha_trained[task['taskId']] = {
            'id': task['taskId'],
            'prompt': task['prompt'],
            'clicks': predicted_clicks,
            'media': media_list
        }
        
        if len(hcaptcha_trained) > 200:
            first_key = list(hcaptcha_trained.keys())[0]
            del hcaptcha_trained[first_key]
            
        return {'solved': True, 'clicks': predicted_clicks}

    return {'solved': False}

@app.route('/api/new-hcaptcha', methods=['POST'])
def new_task():
    task = request.json
    if not task or 'taskId' not in task:
        return jsonify({'success': False})

    auto_res = evaluate_auto_solve(task)
    if auto_res['solved']:
        threading.Thread(target=persist_database).start()
        return jsonify({'success': True, 'autoSolved': True})

    if len(hcaptcha_pending) > 60:
        first_key = list(hcaptcha_pending.keys())[0]
        del hcaptcha_pending[first_key]

    hcaptcha_pending[task['taskId']] = task
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True, 'autoSolved': False})

@app.route('/api/check-hcaptcha/<task_id>', methods=['GET'])
def check_task(task_id):
    if task_id in hcaptcha_trained:
        return jsonify({'status': 'solved', 'clicks': hcaptcha_trained[task_id].get('clicks', [])})
    return jsonify({'status': 'pending'})

@app.route('/api/get-hcaptcha', methods=['GET'])
def get_tasks():
    return jsonify({'pending': hcaptcha_pending, 'trained': hcaptcha_trained})

@app.route('/api/submit-hcaptcha', methods=['POST'])
def submit_task():
    data = request.json
    task_id = data.get('taskId')
    clicks = data.get('clicks', [])
    
    source = hcaptcha_pending.get(task_id) or hcaptcha_trained.get(task_id)
    if source:
        hcaptcha_trained[task_id] = {
            'id': task_id,
            'prompt': source.get('prompt'),
            'media': source.get('media'),
            'clicks': clicks
        }
        if task_id in hcaptcha_pending:
            del hcaptcha_pending[task_id]
        threading.Thread(target=persist_database).start()
        
    return jsonify({'success': True})

@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    if task_id in hcaptcha_pending: del hcaptcha_pending[task_id]
    if task_id in hcaptcha_trained: del hcaptcha_trained[task_id]
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
