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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')
db_lock = threading.Lock()

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
        ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ai_model.eval()
        ai_status = "Ready"
        print("✅ AI Ready!")
    except Exception as e:
        ai_status = "Failed"
        print(f"AI Error: {e}")

threading.Thread(target=load_ai_model, daemon=True).start()

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

def decode_image(src):
    """Base64 ya URL se PIL Image banao"""
    try:
        if src.startswith('data:'):
            b64 = src.split(',')[1]
        else:
            b64 = src
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except:
        return None

def run_ai(task):
    """
    Returns (clicks, confidences) or (None, None)
    
    ✅ FIX: Pehle saari images ko score karo, phir TOP scores wali select karo
    CLIP ko fixed threshold se mat rokao - relative ranking use karo
    """
    if ai_status != "Ready" or not task.get('media'):
        return None, None
    try:
        prompt_full = task.get('prompt', '')
        media = task.get('media', [])

        has_ref = '|||' in prompt_full
        raw_text = prompt_full.split('|||')[0].strip()
        clean_text = (raw_text.lower()
            .replace('select all ', '')
            .replace('please click on the ', '')
            .replace('images with ', '')
            .replace('click on all concepts shown in the reference', '')
            .replace('find all animals based on the number provided', '')
            .strip()) or "object"

        # Images load karo
        loaded = []
        for m in media:
            if not m.get('src'): continue
            img = decode_image(m['src'])
            if img: loaded.append((m['index'], img))

        if not loaded: return None, None

        scores = {}  # index -> score (0-1)

        with torch.no_grad():
            if has_ref:
                # ✅ Image-to-Image: reference se compare
                ref_src = prompt_full.split('|||')[1]
                ref_img = decode_image(ref_src)
                if not ref_img: return None, None

                ref_in = ai_processor(images=ref_img, return_tensors="pt")
                ref_feat = ai_model.get_image_features(**ref_in)
                ref_feat = ref_feat / ref_feat.norm(p=2, dim=-1, keepdim=True)

                for idx, img in loaded:
                    img_in = ai_processor(images=img, return_tensors="pt")
                    feat = ai_model.get_image_features(**img_in)
                    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                    sim = (ref_feat @ feat.T).item()
                    scores[idx] = round(sim, 3)

                # ✅ Relative: mean se upar wali images select karo
                if scores:
                    vals = list(scores.values())
                    mean_score = sum(vals) / len(vals)
                    # Threshold: mean + 10% gap, ya minimum 0.55
                    threshold = max(mean_score + 0.05, 0.55)
                    clicks = [idx for idx, s in scores.items() if s >= threshold]

                    # ✅ Agar koi nahi mila, sirf top 1-2 le lo
                    if not clicks:
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        clicks = [sorted_scores[0][0]] if sorted_scores else []

            else:
                # ✅ Text-to-Image: har image ko score karo
                labels = [
                    f"a photo of {clean_text}",
                    f"a photo of a {clean_text}",
                    "a photo of something completely different",
                ]
                text_in = ai_processor(text=labels, return_tensors="pt", padding=True)

                for idx, img in loaded:
                    img_in = ai_processor(images=img, return_tensors="pt")
                    out = ai_model(**img_in, **text_in)
                    probs = out.logits_per_image.softmax(dim=1)
                    # Score = probability of being the target (label 0 + label 1 combined)
                    score = probs[0][0].item() + probs[0][1].item() * 0.5
                    scores[idx] = round(score, 3)

                # ✅ Relative selection: mean se upar wali
                if scores:
                    vals = list(scores.values())
                    mean_score = sum(vals) / len(vals)
                    max_score  = max(vals)

                    # Dynamic threshold
                    # Agar max score bohot high hai (>0.6), strict threshold
                    # Agar max score medium hai (0.3-0.6), loose threshold
                    if max_score > 0.6:
                        threshold = mean_score + 0.08
                    elif max_score > 0.35:
                        threshold = mean_score + 0.03
                    else:
                        # Scores bohot low - shayad AI nahi samajh raha, null return karo
                        return None, {}

                    clicks = [idx for idx, s in scores.items() if s >= threshold]

                    # ✅ Minimum 1, maximum based on typical hCaptcha (1-6 usually)
                    if not clicks and max_score > 0.25:
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        clicks = [sorted_scores[0][0]]

        gc.collect()
        confidences = {str(k): v for k, v in scores.items()}
        print(f"AI Result: clicks={clicks}, scores={scores}")
        return clicks, confidences

    except Exception as e:
        print(f"AI Error: {e}")
        return None, None


def ai_background_job(tid):
    task = hcaptcha_unsolved.get(tid)
    if not task: return

    ai_clicks, confidences = run_ai(task)

    if tid not in hcaptcha_unsolved: return  # manual ne solve kar diya

    if ai_clicks is not None:
        hcaptcha_trained[tid] = {
            'id': tid,
            'prompt': task.get('prompt', ''),
            'media': task.get('media', []),
            'clicks': ai_clicks,
            'ai_clicks': ai_clicks,
            'source': 'ai',
            'confidences': confidences or {},
            'timestamp': time.time()
        }
        hcaptcha_unsolved.pop(tid, None)
        print(f"✅ AI solved {tid}: {ai_clicks}")
    else:
        print(f"⏳ AI could not solve {tid}, manual needed")

    threading.Thread(target=save_db, daemon=True).start()


@app.route('/', methods=['GET'])
def home():
    return jsonify({'status':'Active','ai':ai_status,'unsolved':len(hcaptcha_unsolved),'trained':len(hcaptcha_trained)})

@app.route('/api/new-hcaptcha', methods=['POST','OPTIONS'])
def new_task():
    if request.method=='OPTIONS': return jsonify({'success':True}),200
    task = request.json
    if not task or 'taskId' not in task: return jsonify({'success':False})
    tid = task['taskId']

    # Already trained?
    if tid in hcaptcha_trained:
        entry = hcaptcha_trained[tid]
        clicks = entry.get('clicks',[])
        if clicks:
            return jsonify({'success':True,'autoSolved':True,'clicks':clicks,'source':entry.get('source')})

    # Already processing?
    if tid in hcaptcha_unsolved:
        return jsonify({'success':True,'autoSolved':False,'status':'processing'})

    # Naya task - PEHLE unsolved mein save karo
    if len(hcaptcha_unsolved) >= 150:
        oldest = min(hcaptcha_unsolved, key=lambda k: hcaptcha_unsolved[k].get('timestamp',0))
        del hcaptcha_unsolved[oldest]

    hcaptcha_unsolved[tid] = {
        'id': tid,
        'prompt': task.get('prompt',''),
        'media': task.get('media',[]),
        'clicks': [],
        'source': 'unsolved',
        'timestamp': time.time()
    }
    threading.Thread(target=save_db, daemon=True).start()

    # PHIR background mein AI
    threading.Thread(target=ai_background_job, args=(tid,), daemon=True).start()

    return jsonify({'success':True,'autoSolved':False,'ai_status':ai_status})

@app.route('/api/check-hcaptcha/<task_id>', methods=['GET','OPTIONS'])
def check_task(task_id):
    if request.method=='OPTIONS': return jsonify({'success':True}),200
    if task_id in hcaptcha_trained:
        entry = hcaptcha_trained[task_id]
        clicks = entry.get('clicks',[])
        if clicks:
            return jsonify({'status':'solved','clicks':clicks,'source':entry.get('source')})
    return jsonify({'status':'pending'})

@app.route('/api/get-hcaptcha', methods=['GET','OPTIONS'])
def get_tasks():
    if request.method=='OPTIONS': return jsonify({'success':True}),200
    return jsonify({'unsolved':hcaptcha_unsolved,'trained':hcaptcha_trained,'ai_status':ai_status})

@app.route('/api/submit-hcaptcha', methods=['POST','OPTIONS'])
def submit_task():
    if request.method=='OPTIONS': return jsonify({'success':True}),200
    data = request.json
    tid = data.get('taskId')
    clicks = data.get('clicks',[])
    is_retrain = data.get('retrain', False)
    if not tid: return jsonify({'success':False})

    existing = hcaptcha_trained.get(tid,{})
    old_source = existing.get('source','')
    new_source = 'retrained' if (is_retrain and old_source=='ai') else 'manual'
    base = hcaptcha_unsolved.get(tid) or existing

    hcaptcha_trained[tid] = {
        'id': tid,
        'prompt': base.get('prompt',''),
        'media': base.get('media',[]),
        'clicks': clicks,
        'ai_clicks': existing.get('ai_clicks', existing.get('clicks',[])),
        'source': new_source,
        'confidences': existing.get('confidences',{}),
        'timestamp': time.time()
    }
    hcaptcha_unsolved.pop(tid, None)
    threading.Thread(target=save_db, daemon=True).start()
    return jsonify({'success':True,'source':new_source})

@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE','OPTIONS'])
def delete_task(task_id):
    if request.method=='OPTIONS': return jsonify({'success':True}),200
    hcaptcha_unsolved.pop(task_id, None)
    hcaptcha_trained.pop(task_id, None)
    threading.Thread(target=save_db, daemon=True).start()
    return jsonify({'success':True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
