import os
import json
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# AI Models imports
try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    import requests
    from io import BytesIO
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: transformers/torch not installed. AI features disabled.")

app = Flask(__name__)
# CORS error hamesha ke liye khatam
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')

hcaptcha_pending = {}
hcaptcha_trained = {}

# Global AI Variables
ai_model = None
ai_processor = None
ai_status = "Not Loaded"

def load_ai_model():
    global ai_model, ai_processor, ai_status
    if not AI_AVAILABLE:
        ai_status = "Failed (Dependencies missing)"
        return
        
    try:
        print("Starting Container AI Logic...")
        print("Loading CLIP Model (For High Accuracy Grid Classification)...")
        ai_status = "Loading weights..."
        
        # Load your heavy AI model here
        ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        ai_status = "Ready"
        print("Loading weights: 100% - AI Engine is completely LOADED and ACTIVE!")
    except Exception as e:
        print(f"AI Load Error: {e}")
        ai_status = "Failed"

# Railway timeout bypass: Start AI in background!
threading.Thread(target=load_ai_model).start()

if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, 'r') as f:
            data = json.load(f)
            hcaptcha_trained = data.get('trained', {})
    except Exception as e:
        pass

def persist_database():
    try:
        with open(DB_FILE, 'w') as f:
            json.dump({'pending': hcaptcha_pending, 'trained': hcaptcha_trained}, f)
    except:
        pass

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': f'Server is running perfectly on Port 3000! AI Status: {ai_status}'})

@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    
    task = request.json
    if not task or 'taskId' not in task:
        return jsonify({'success': False})

    # 1. Check if we already trained it manually
    if task['taskId'] in hcaptcha_trained:
        return jsonify({'success': True, 'autoSolved': True, 'clicks': hcaptcha_trained[task['taskId']].get('clicks', [])})

    # 2. FULL AI AUTO-SOLVER LOGIC
    if ai_status == "Ready":
        # Yahan aapki AI automatically prompt parh kar images select karegi
        # (Yeh ek structural example hai, model actual prediction yahan return karega)
        print(f"AI Processing Task ID: {task['taskId']} | Prompt: {task.get('prompt')}")
        
        # Example Auto-Click Logic by AI:
        # ai_clicks = [0, 2, 4] # AI decide karegi konsi tasweer click karni hai
        # hcaptcha_trained[task['taskId']] = {'clicks': ai_clicks}
        # return jsonify({'success': True, 'autoSolved': True, 'clicks': ai_clicks})

    # 3. Agar AI abhi load ho rahi hai ya new image samajh nahi aayi toh Pending mein daal do
    if len(hcaptcha_pending) > 60:
        first_key = list(hcaptcha_pending.keys())[0]
        del hcaptcha_pending[first_key]

    hcaptcha_pending[task['taskId']] = task
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True, 'autoSolved': False, 'ai_status': ai_status})

@app.route('/api/get-hcaptcha', methods=['GET', 'OPTIONS'])
def get_tasks():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    return jsonify({'pending': hcaptcha_pending, 'trained': hcaptcha_trained, 'ai_status': ai_status})

@app.route('/api/submit-hcaptcha', methods=['POST', 'OPTIONS'])
def submit_task():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
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

@app.route('/api/delete-hcaptcha/<task_id>', methods=['DELETE', 'OPTIONS'])
def delete_task(task_id):
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    if task_id in hcaptcha_pending: del hcaptcha_pending[task_id]
    if task_id in hcaptcha_trained: del hcaptcha_trained[task_id]
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
