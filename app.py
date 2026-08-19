import os
import json
import threading
import base64
import time
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc

try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI libs not installed.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')
db_lock = threading.Lock()

# Sirf 2 dicts:
# unsolved  = AI fail ya naya task, dashboard mein dikhta hai
# trained   = AI ya manual se solve hua
hcaptcha_unsolved = {}
hcaptcha_trained  = {}

ai_model = None
ai_processor = None
ai_status = "Loading..."

def load_ai_model():
    global ai_model, ai_processor, ai_status
    if not AI_AVAILABLE:
        ai_status = "Failed"; return
    try:
        print("Loading CLIP...")
        ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ai_model.eval()
        ai_status = "Ready"
        print("✅ AI Ready!")
    except Exception as e:
        ai_status = "Failed"
        print(f"AI Load Error: {e}")

threading.Thread(target=load_ai_model, daemon=True).start()

# DB load
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            hcaptcha_unsolved = data.get('unsolved', {})
            hcaptcha_trained  = data.get('trained', {})
    except: pass

def save_db():
    with db_lock:
        try:
            tmp = DB_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({'unsolved': hcaptcha_unsolved, 'trained': hcaptcha_trained}, f)
            os.replace(tmp, DB_FILE)
        except Exception as e:
            print(f"DB Error: {e}")

def run_ai(task):
    """Returns (clicks, confidences) or (None, None)"""
    if ai_status != "Ready" or not task.get('media'):
        return None, None
    try:
        prompt_full = task.get('prompt', '')
        media = task.get('media', [])
        clicks = []
        confidences = {}

        has_ref = '|||' in prompt_full
        raw_text = prompt_full.split('|||')[0].strip()
        clean_text = (raw_text.lower()
            .replace('select all ', '')
            .replace('please click on the ', '')
            .replace('images with ', '')
            .replace('click on all concepts shown in the reference', '')
            .strip()) or "object"

        target_images = []
        for m in media:
            if not m.get('src'): continue
            try:
                b64 = m['src'].split(',')[1] if ',' in m['src'] else m['src']
                img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                target_images.append((m['index'], img))
            except: continue

        if not target_images: return None, None

        with torch.no_grad():
            if has_ref:
                ref_b64 = prompt_full.split('|||')[1].split(',')[-1]
                ref_img = Image.open(BytesIO(base64.b64decode(ref_b64))).convert("RGB")
                ref_in = ai_processor(images=ref_img, return_tensors="pt")
                ref_feat = ai_model.get_image_features(**ref_in)
                ref_feat = ref_feat / ref_feat.norm(p=2, dim=-1, keepdim=True)

                for idx, img in target_images:
                    inp = ai_processor(images=img, return_tensors="pt")
                    feat = ai_model.get_image_features(**inp)
                    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                    sim = (ref_feat @ feat.T).item()
                    confidences[str(idx)] = round(sim, 3)
                    if sim > 0.70: clicks.append(idx)
            else:
                labels = [f"a photo of {clean_text}", "a photo of an unrelated object", "a photo of scenery"]
                text_in = ai_processor(text=labels, return_tensors="pt", padding=True)
                for idx, img in target_images:
                    img_in = ai_processor(images=img, return_tensors="pt")
                    out = ai_model(**img_in, **text_in)
                    probs = out.logits_per_image.softmax(dim=1)
                    score = probs[0][0].item()
                    confidences[str(idx)] = round(score, 3)
                    if score > 0.40: clicks.append(idx)

        gc.collect()
        return clicks, confidences
    except Exception as e:
        print(f"AI Error: {e}")
        return None, None


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Active',
        'ai': ai_status,
        'unsolved': len(hcaptcha_unsolved),
        'trained': len(hcaptcha_trained)
    })

@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    task = request.json
    if not task or 'taskId' not in task: return jsonify({'success': False})
    tid = task['taskId']

    # ✅ Pehle trained check karo - seedha solve
    if tid in hcaptcha_trained:
        entry = hcaptcha_trained[tid]
        clicks = entry.get('clicks', [])
        # Sirf tab autoSolved = True jab clicks hain
        if clicks:
            return jsonify({'success': True, 'autoSolved': True, 'clicks': clicks, 'source': entry.get('source', 'manual')})

    # ✅ AI run karo
    ai_clicks, confidences = run_ai(task)

    if ai_clicks is not None:
        # AI kaamyab - seedha trained mein daalo, unsolved mein NAHI
        hcaptcha_trained[tid] = {
            'id': tid,
            'prompt': task.get('prompt', ''),
            'media': task.get('media', []),
            'clicks': ai_clicks,
            'ai_clicks': ai_clicks,       # original AI clicks (retrain ke liye)
            'source': 'ai',
            'confidences': confidences or {},
            'timestamp': time.time()
        }
        threading.Thread(target=save_db, daemon=True).start()
        return jsonify({'success': True, 'autoSolved': True, 'clicks': ai_clicks, 'source': 'ai'})

    # ✅ AI fail - unsolved mein daalo, extension polling karega
    if len(hcaptcha_unsolved) >= 100:
        oldest = min(hcaptcha_unsolved, key=lambda k: hcaptcha_unsolved[k].get('timestamp', 0))
        del hcaptcha_unsolved[oldest]
    hcaptcha_unsolved[tid] = {
        'id': tid,
        'prompt': task.get('prompt', ''),
        'media': task.get('media', []),
        'clicks': [],
        'source': 'unsolved',
        'timestamp': time.time()
    }
    threading.Thread(target=save_db, daemon=True).start()
    return jsonify({'success': True, 'autoSolved': False, 'ai_status': ai_status})


@app.route('/api/check-hcaptcha/<task_id>', methods=['GET', 'OPTIONS'])
def check_task(task_id):
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200

    # ✅ KEY FIX: trained mein dekho with actual clicks
    if task_id in hcaptcha_trained:
        entry = hcaptcha_trained[task_id]
        clicks = entry.get('clicks', [])
        if clicks:  # sirf tab solved jab clicks actually hain
            return jsonify({'status': 'solved', 'clicks': clicks, 'source': entry.get('source')})

    # Abhi bhi unsolved
    return jsonify({'status': 'pending'})


@app.route('/api/get-hcaptcha', methods=['GET', 'OPTIONS'])
def get_tasks():
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    return jsonify({
        'unsolved': hcaptcha_unsolved,
        'trained': hcaptcha_trained,
        'ai_status': ai_status
    })


@app.route('/api/submit-hcaptcha', methods=['POST', 'OPTIONS'])
def submit_task():
    """Dashboard se clicks submit karo"""
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    data = request.json
    tid = data.get('taskId')
    clicks = data.get('clicks', [])
    is_retrain = data.get('retrain', False)
    if not tid: return jsonify({'success': False})

    # Source kya hoga
    existing_trained = hcaptcha_trained.get(tid, {})
    old_source = existing_trained.get('source', '')
    if is_retrain and old_source == 'ai':
        new_source = 'retrained'
    else:
        new_source = 'manual'

    # unsolved se base data lo
    base = hcaptcha_unsolved.get(tid) or existing_trained

    hcaptcha_trained[tid] = {
        'id': tid,
        'prompt': base.get('prompt', ''),
        'media': base.get('media', []),
        'clicks': clicks,
        'ai_clicks': existing_trained.get('ai_clicks', existing_trained.get('clicks', [])),
        'source': new_source,
        'confidences': existing_trained.get('confidences', {}),
        'timestamp': time.time()
    }
    # unsolved se remove karo
    hcaptcha_unsolved.pop(tid, None)
    threading.Thread(target=save_db, daemon=True).start()
    return jsonify({'success': True, 'source': new_source})


@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE', 'OPTIONS'])
def delete_task(task_id):
    if request.method == 'OPTIONS': return jsonify({'success': True}), 200
    hcaptcha_unsolved.pop(task_id, None)
    hcaptcha_trained.pop(task_id, None)
    threading.Thread(target=save_db, daemon=True).start()
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
