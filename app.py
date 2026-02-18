from flask import Flask, render_template, request, redirect, url_for, session
import os
import pickle
import numpy as np

# Load or train model/scaler
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception:
    # Lazy-train if missing
    from train_model import train_and_save
    train_and_save(csv_path=os.path.join(os.path.dirname(__file__), 'anemia.csv'), model_dir=MODEL_DIR)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'change-this-secret')

# Simple credentials (demo only) - replace with real auth in production
VALID_USER = 'admin'
VALID_PASS = '1234'

@app.route('/')
def index():
    if session.get('user'):
        return redirect(url_for('predict'))
    error = request.args.get('error')
    return render_template('index.html', error=error)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if username == VALID_USER and password == VALID_PASS:
        session['user'] = username
        return redirect(url_for('predict'))
    return redirect(url_for('index', error=1))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('user'):
        return redirect(url_for('index'))

    result = None
    data = {}
    if request.method == 'POST':
        name = request.form.get('name', '')
        # Expect gender as 1 or 0
        try:
            gender = int(request.form.get('gender', 0))
        except ValueError:
            gender = 0
        def to_float(field):
            try:
                return float(request.form.get(field, 0))
            except Exception:
                return 0.0
        hemoglobin = to_float('hemoglobin')
        mch = to_float('mch')
        mchc = to_float('mchc')
        mcv = to_float('mcv')

        # Create input vector in the same column order as training CSV: Gender,Hemoglobin,MCH,MCHC,MCV
        x = np.array([[gender, hemoglobin, mch, mchc, mcv]], dtype=float)
        x_scaled = scaler.transform(x)
        prob = model.predict_proba(x_scaled)[0]
        pred = 1 if prob[1] > 0.01 else 0  # Lower threshold
        anemia = 'Likely' if int(pred) == 1 else 'Unlikely'
        result = {
            'name': name,
            'gender': gender,
            'hemoglobin': hemoglobin,
            'mcv': mcv,
            'mch': mch,
            'mchc': mchc,
            'anemia': anemia
        }
        data = result

    return render_template('predict.html', result=result, data=data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
