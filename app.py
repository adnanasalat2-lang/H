import os
import json
import threading
import base64
import time
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

# Sirf ek dictionary - sab trained mein jata hai
# har entry mein 'source': 'ai' ya 'manual' hoga
hcaptcha_trained = {}

# File write ke liye lock (race condition se bachao)
db_lock = threading.Lock()

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
        ai_model.eval()
        ai_status = "Ready"
        print("AI Engine LOADED!")
    except Exception as e:
        ai_status = "Failed"
        print(f"AI Load Error: {e}")

threading.Thread(target=load_ai_model, daemon=True).start()

# DB load on startup
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            hcaptcha_trained = data.get('trained', {})
    except Exception as e:
        print(f"DB Load Error: {e}")

def persist_database():
    with db_lock:
        try:
            tmp_file = DB_FILE + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump({'trained': hcaptcha_trained}, f)
            os.replace(tmp_file, DB_FILE)
        except Exception as e:
            print(f"DB Save Error: {e}")

def process_ai_task(task):
    """AI se clicks predict karo. Returns (clicks, confidence_list) ya (None, None)"""
    if ai_status != "Ready" or not task.get('media'):
        return None, None
    try:
        prompt_full = task.get('prompt', '')
        media = task.get('media', [])
        clicks = []
        confidences = {}  # index -> score

        has_ref = '|||' in prompt_full
        raw_text = prompt_full.split('|||')[0].strip()

        clean_text = raw_text.lower()\
            .replace('select all ', '')\
            .replace('please click on the ', '')\
            .replace('images with ', '')\
            .replace('click on all concepts shown in the reference', '')\
            .strip()
        if not clean_text:
            clean_text = "object"

        target_images = []
        for m in media:
            if not m.get('src'):
                continue
            try:
                m_b64 = m['src'].split(',')[1] if ',' in m['src'] else m['src']
                img = Image.open(BytesIO(base64.b64decode(m_b64))).convert("RGB")
                target_images.append((m['index'], img))
            except Exception:
                continue

        if not target_images:
            return None, None

        with torch.no_grad():
            if has_ref:
                # Image-to-Image Matching
                ref_part = prompt_full.split('|||')[1]
                ref_b64 = ref_part.split(',')[-1]
                ref_img = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGB")

                ref_inputs = ai_processor(images=ref_img, return_tensors="pt")
                ref_features = ai_model.get_image_features(**ref_inputs)
                ref_features = ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)

                for idx, img in target_images:
                    img_inputs = ai_processor(images=img, return_tensors="pt")
                    img_features = ai_model.get_image_features(**img_inputs)
                    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                    similarity = (ref_features @ img_features.T).item()
                    confidences[idx] = round(similarity, 3)
                    if similarity > 0.70:
                        clicks.append(idx)
            else:
                # Text-to-Image Matching
                candidate_labels = [
                    f"a photo of {clean_text}",
                    "a photo of an unrelated object",
                    "a photo of a background or scenery"
                ]
                text_inputs = ai_processor(text=candidate_labels, return_tensors="pt", padding=True)

                for idx, img in target_images:
                    img_inputs = ai_processor(images=img, return_tensors="pt")
                    outputs = ai_model(**img_inputs, **text_inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    score = probs[0][0].item()
                    confidences[idx] = round(score, 3)
                    if score > 0.40:
                        clicks.append(idx)

        gc.collect()
        return clicks, confidences

    except Exception as e:
        print(f"AI Processing Error: {e}")
        return None, None


@app.route('/', methods=['GET'])
def home():
    total = len(hcaptcha_trained)
    ai_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'ai')
    manual_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'manual')
    retrained_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'retrained')
    return jsonify({
        'status': f'Server Active | AI: {ai_status}',
        'total_trained': total,
        'ai_solved': ai_count,
        'manual_solved': manual_count,
        'retrained': retrained_count
    })


@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200

    task = request.json
    if not task or 'taskId' not in task:
        return jsonify({'success': False, 'error': 'Invalid task'})

    task_id = task['taskId']

    # Pehle check: pehle se trained hai?
    if task_id in hcaptcha_trained:
        entry = hcaptcha_trained[task_id]
        return jsonify({
            'success': True,
            'autoSolved': True,
            'clicks': entry.get('clicks', []),
            'source': entry.get('source', 'manual'),
            'confidences': entry.get('confidences', {})
        })

    # AI se solve karne ki koshish
    ai_clicks, confidences = process_ai_task(task)

    # Dono cases mein trained mein save karo (media bhi save karo dashboard ke liye)
    if ai_clicks is not None:
        # AI ne solve kiya
        hcaptcha_trained[task_id] = {
            'id': task_id,
            'prompt': task.get('prompt', ''),
            'media': task.get('media', []),
            'clicks': ai_clicks,
            'source': 'ai',
            'confidences': confidences or {},
            'timestamp': time.time()
        }
        threading.Thread(target=persist_database, daemon=True).start()
        return jsonify({
            'success': True,
            'autoSolved': True,
            'clicks': ai_clicks,
            'source': 'ai',
            'confidences': confidences or {}
        })
    else:
        # AI fail - phir bhi trained mein save karo as 'pending_manual'
        # Extension poling karta rahega jab tak manual submit na ho
        hcaptcha_trained[task_id] = {
            'id': task_id,
            'prompt': task.get('prompt', ''),
            'media': task.get('media', []),
            'clicks': [],
            'source': 'pending_manual',
            'confidences': {},
            'timestamp': time.time(),
            'ai_status': ai_status
        }
        threading.Thread(target=persist_database, daemon=True).start()
        return jsonify({
            'success': True,
            'autoSolved': False,
            'source': 'pending_manual',
            'ai_status': ai_status
        })


@app.route('/api/check-hcaptcha/<task_id>', methods=['GET', 'OPTIONS'])
def check_task(task_id):
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200

    if task_id in hcaptcha_trained:
        entry = hcaptcha_trained[task_id]
        source = entry.get('source', 'manual')

        # Agar pending_manual hai aur clicks aa gaye hain to solved
        if source == 'pending_manual' and len(entry.get('clicks', [])) == 0:
            return jsonify({'status': 'pending'})

        return jsonify({
            'status': 'solved',
            'clicks': entry.get('clicks', []),
            'source': source,
            'confidences': entry.get('confidences', {})
        })

    return jsonify({'status': 'not_found'})


@app.route('/api/get-hcaptcha', methods=['GET', 'OPTIONS'])
def get_tasks():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    return jsonify({
        'trained': hcaptcha_trained,
        'ai_status': ai_status
    })


@app.route('/api/submit-hcaptcha', methods=['POST', 'OPTIONS'])
def submit_task():
    """Dashboard se manual clicks submit karo (naya ya re-train)"""
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200

    data = request.json
    task_id = data.get('taskId')
    clicks = data.get('clicks', [])
    is_retrain = data.get('retrain', False)

    if not task_id:
        return jsonify({'success': False, 'error': 'No taskId'})

    existing = hcaptcha_trained.get(task_id, {})
    old_source = existing.get('source', 'unknown')

    # Source determine karo
    if is_retrain and old_source == 'ai':
        new_source = 'retrained'  # AI ki galti ko fix kiya
    elif old_source == 'pending_manual':
        new_source = 'manual'
    else:
        new_source = 'manual'

    hcaptcha_trained[task_id] = {
        'id': task_id,
        'prompt': existing.get('prompt', data.get('prompt', '')),
        'media': existing.get('media', data.get('media', [])),
        'clicks': clicks,
        'source': new_source,
        'confidences': existing.get('confidences', {}),
        'ai_clicks': existing.get('clicks', []) if old_source == 'ai' else existing.get('ai_clicks', []),
        'timestamp': time.time()
    }

    threading.Thread(target=persist_database, daemon=True).start()
    return jsonify({'success': True, 'source': new_source})


@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE', 'OPTIONS'])
def delete_task(task_id):
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    hcaptcha_trained.pop(task_id, None)
    threading.Thread(target=persist_database, daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    total = len(hcaptcha_trained)
    ai_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'ai')
    manual_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'manual')
    retrained_count = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'retrained')
    pending_manual = sum(1 for v in hcaptcha_trained.values() if v.get('source') == 'pending_manual')
    return jsonify({
        'total': total,
        'ai': ai_count,
        'manual': manual_count,
        'retrained': retrained_count,
        'pending_manual': pending_manual,
        'ai_status': ai_status
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
