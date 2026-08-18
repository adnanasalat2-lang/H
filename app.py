import os
import json
import threading
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc

# AI Models
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
        print("Downloading & Loading AI CLIP Model...")
        ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ai_model.eval()  # Model ko evaluation mode mein daalna taake RAM bachay
        ai_status = "Ready"
        print("✅ AI Engine 100% LOADED, ACTIVE & RAM-SAFE!")
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
    if ai_status != "Ready" or not task.get('media'): return None
    try:
        prompt_full = task.get('prompt', '')
        media = task.get('media', [])
        clicks = []

        has_ref = '|||' in prompt_full
        raw_text = prompt_full.split('|||')[0].strip()
        
        # Kolotibablo ke prompts ki perfect cleaning
        clean_text = raw_text.lower().replace('select all ', '').replace('please click on the ', '').replace('images with ', '').replace('click on all concepts shown in the reference', '')
        if clean_text.strip() == "": clean_text = "object"

        target_images = []
        for m in media:
            m_b64 = m['src'].split(',')[1] if ',' in m['src'] else m['src']
            img = Image.open(BytesIO(base64.b64decode(m_b64))).convert("RGB")
            target_images.append((m['index'], img))

        # 🚀 torch.no_grad() is the magic lock jo RAM ko hamesha khali rakhega aur crash nahi hone dega
        with torch.no_grad():
            if has_ref:
                # Image-to-Image Matching
                ref_b64 = prompt_full.split('|||')[1].split(',')[-1]
                ref_img = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGB")
                
                ref_inputs = ai_processor(images=ref_img, return_tensors="pt")
                ref_features = ai_model.get_image_features(**ref_inputs)
                ref_features = ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)

                for idx, img in target_images:
                    img_inputs = ai_processor(images=img, return_tensors="pt")
                    img_features = ai_model.get_image_features(**img_inputs)
                    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                    
                    similarity = (ref_features @ img_features.T).item()
                    if similarity > 0.70:
                        clicks.append(idx)
            else:
                # Text-to-Image Matching (Fridge, Strawberry etc)
                candidate_labels = [f"a photo of {clean_text}", "a photo of an unrelated object", "a photo of a background or scenery"]
                text_inputs = ai_processor(text=candidate_labels, return_tensors="pt", padding=True)

                for idx, img in target_images:
                    img_inputs = ai_processor(images=img, return_tensors="pt")
                    outputs = ai_model(**img_inputs, **text_inputs)
                    probs = outputs.logits_per_image.softmax(dim=1) 
                    
                    if probs[0][0].item() > 0.40:
                        clicks.append(idx)

        # Cache clear karo taake 48 profiles mein bhi heavy na ho
        gc.collect()
        return clicks
    except Exception as e:
        print(f"AI Processing Error: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': f'Paid Server Active on Port 3000! AI: {ai_status}'})

@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    task = request.json
    if not task or 'taskId' not in task: return jsonify({'success': False})

    # Manual Train Check
    if task['taskId'] in hcaptcha_trained:
        return jsonify({'success': True, 'autoSolved': True, 'clicks': hcaptcha_trained[task['taskId']].get('clicks', [])})

    # Asli AI Engine
    ai_clicks = process_ai_task(task)
    if ai_clicks is not None and len(ai_clicks) > 0:
        hcaptcha_trained[task['taskId']] = {'clicks': ai_clicks}
        return jsonify({'success': True, 'autoSolved': True, 'clicks': ai_clicks})

    # AI Fail hone par dashboard par
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
