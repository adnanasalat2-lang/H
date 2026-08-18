import os
import json
import threading
import time
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS

# AI Models (Background load honge)
try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: transformers/torch not installed.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')

hcaptcha_pending = {}
hcaptcha_trained = {}

ai_model = None
ai_processor = None
ai_status = "Loading..."

def load_ai_model():
    global ai_model, ai_processor, ai_status
    if not AI_AVAILABLE:
        ai_status = "Failed"
        return
    try:
        print("Downloading & Loading AI CLIP Model in Background...")
        ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ai_status = "Ready"
        print("✅ AI Engine 100% LOADED & ACTIVE!")
    except Exception as e:
        ai_status = "Failed"
        print(f"AI Load Error: {e}")

threading.Thread(target=load_ai_model).start()

if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            hcaptcha_trained = data.get('trained', {})
    except Exception as e: pass

def persist_database():
    try:
        with open(DB_FILE, 'w') as f:
            json.dump({'pending': hcaptcha_pending, 'trained': hcaptcha_trained}, f)
    except: pass

def process_ai_task(task):
    """Mukammal AI Engine: Image-to-Image aur Text-to-Image dono ke liye"""
    if ai_status != "Ready" or not task.get('media'): return None
    try:
        prompt_full = task.get('prompt', '')
        media = task.get('media', [])
        clicks = []

        # Check karein ke reference image hai ya sirf text hai
        has_ref = '|||' in prompt_full
        text_prompt = prompt_full.split('|||')[0].strip()

        # Grid tasweeron ko process karein
        target_images = []
        for m in media:
            m_b64 = m['src'].split(',')[1] if ',' in m['src'] else m['src']
            img = Image.open(BytesIO(base64.b64decode(m_b64))).convert("RGB")
            target_images.append((m['index'], img))

        if has_ref:
            # SCENARIO 1: IMAGE-TO-IMAGE MATCHING
            ref_b64 = prompt_full.split('|||')[1]
            ref_b64 = ref_b64.split(',')[1] if ',' in ref_b64 else ref_b64
            ref_img = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGB")
            
            ref_inputs = ai_processor(images=ref_img, return_tensors="pt")
            ref_features = ai_model.get_image_features(**ref_inputs)
            ref_features = ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)

            for idx, img in target_images:
                img_inputs = ai_processor(images=img, return_tensors="pt")
                img_features = ai_model.get_image_features(**img_inputs)
                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                
                similarity = (ref_features @ img_features.T).item()
                if similarity > 0.85:
                    clicks.append(idx)
        else:
            # SCENARIO 2: TEXT-TO-IMAGE MATCHING (Jaise fridge wali tasveer)
            clip_text = f"a photo of {text_prompt}"
            text_inputs = ai_processor(text=[clip_text], return_tensors="pt", padding=True)
            text_features = ai_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            for idx, img in target_images:
                img_inputs = ai_processor(images=img, return_tensors="pt")
                img_features = ai_model.get_image_features(**img_inputs)
                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                
                similarity = (text_features @ img_features.T).item()
                # CLIP mein text-to-image similarity aam tor par 0.22 se oopar match hoti hai
                if similarity > 0.22:
                    clicks.append(idx)
                    
        return clicks
    except Exception as e:
        print(f"AI Processing Error: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': f'Server Running on Port 3000! AI: {ai_status}'})

@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    task = request.json
    if not task or 'taskId' not in task: return jsonify({'success': False})

    # 1. Manual Cache check
    if task['taskId'] in hcaptcha_trained:
        return jsonify({'success': True, 'autoSolved': True, 'clicks': hcaptcha_trained[task['taskId']].get('clicks', [])})

    # 2. FULL AI AUTO-SOLVE (Ab dono text aur image ke liye)
    ai_clicks = process_ai_task(task)
    if ai_clicks is not None and len(ai_clicks) > 0:
        hcaptcha_trained[task['taskId']] = {'clicks': ai_clicks}
        return jsonify({'success': True, 'autoSolved': True, 'clicks': ai_clicks})

    # 3. Fail hone par Dashboard
    if len(hcaptcha_pending) > 60: del hcaptcha_pending[list(hcaptcha_pending.keys())[0]]
    hcaptcha_pending[task['taskId']] = task
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True, 'autoSolved': False, 'ai_status': ai_status})

@app.route('/api/check-hcaptcha/<task_id>', methods=['GET', 'OPTIONS'])
def check_task(task_id):
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    if task_id in hcaptcha_trained: return jsonify({'status': 'solved', 'clicks': hcaptcha_trained[task_id].get('clicks', [])})
    return jsonify({'status': 'pending'})

@app.route('/api/get-hcaptcha', methods=['GET', 'OPTIONS'])
def get_tasks():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    return jsonify({'pending': hcaptcha_pending, 'trained': hcaptcha_trained})

@app.route('/api/submit-hcaptcha', methods=['POST', 'OPTIONS'])
def submit_task():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    data = request.json
    task_id = data.get('taskId')
    clicks = data.get('clicks', [])
    source = hcaptcha_pending.get(task_id) or hcaptcha_trained.get(task_id)
    if source:
        hcaptcha_trained[task_id] = {'id': task_id, 'prompt': source.get('prompt'), 'media': source.get('media'), 'clicks': clicks}
        if task_id in hcaptcha_pending: del hcaptcha_pending[task_id]
        threading.Thread(target=persist_database).start()
    return jsonify({'success': True})

@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE', 'OPTIONS'])
def delete_task(task_id):
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    if task_id in hcaptcha_pending: del hcaptcha_pending[task_id]
    if task_id in hcaptcha_trained: del hcaptcha_trained[task_id]
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
