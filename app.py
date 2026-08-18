import os
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Yeh line har qisam ke CORS error ko hamesha ke liye allow kar degi
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = '/data' if os.path.exists('/data') else os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DATA_DIR, 'database.json')

hcaptcha_pending = {}
hcaptcha_trained = {}

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
    return jsonify({'status': 'Server is running perfectly!'})

@app.route('/api/new-hcaptcha', methods=['POST', 'OPTIONS'])
def new_task():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    task = request.json
    if not task or 'taskId' not in task:
        return jsonify({'success': False})

    if task['taskId'] in hcaptcha_trained:
        return jsonify({'success': True, 'autoSolved': True, 'clicks': hcaptcha_trained[task['taskId']].get('clicks', [])})

    if len(hcaptcha_pending) > 60:
        first_key = list(hcaptcha_pending.keys())[0]
        del hcaptcha_pending[first_key]

    hcaptcha_pending[task['taskId']] = task
    threading.Thread(target=persist_database).start()
    return jsonify({'success': True, 'autoSolved': False})

@app.route('/api/check-hcaptcha/<task_id>', methods=['GET', 'OPTIONS'])
def check_task(task_id):
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    if task_id in hcaptcha_trained:
        return jsonify({'status': 'solved', 'clicks': hcaptcha_trained[task_id].get('clicks', [])})
    return jsonify({'status': 'pending'})

@app.route('/api/get-hcaptcha', methods=['GET', 'OPTIONS'])
def get_tasks():
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
    return jsonify({'pending': hcaptcha_pending, 'trained': hcaptcha_trained})

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
    # Railway khud PORT provide karta hai, agar na mile toh 3000
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
