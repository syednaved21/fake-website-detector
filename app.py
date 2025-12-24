from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
from features import extract_url_features, heuristic_score

app = Flask(__name__)

MODEL_PATH = 'model/model.pkl'
SCALER_PATH = 'model/scaler.pkl'

clf = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print('Loaded ML model.')
    except Exception as e:
        print('Failed to load model, using heuristic:', e)

def predict_with_model(url: str):
    feats = pd.DataFrame([extract_url_features(url)])
    X = feats.values
    Xs = scaler.transform(X) if scaler is not None else X
    proba = clf.predict_proba(Xs)[0][1]
    score = int(proba * 100)
    label = 'Fraudulent' if score >= 65 else ('Suspicious' if score >= 35 else 'Safe')
    d = feats.iloc[0].to_dict()
    d['risk_score'] = score
    d['ml_probability'] = float(proba)
    return score, label, d

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    details = None
    url_value = ''
    using_ml = bool(clf is not None)
    if request.method == 'POST':
        url_value = request.form.get('url','').strip()
        if url_value:
            if using_ml:
                score, label, details = predict_with_model(url_value)
            else:
                score, label, details = heuristic_score(url_value)
            result = {'label': label, 'score': score}
    return render_template('index.html', result=result, details=details, url_value=url_value, using_ml=using_ml)

@app.route('/about')
def about():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
