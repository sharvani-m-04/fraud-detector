from flask import Flask, request, render_template
import joblib, json, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
lr_model = joblib.load(os.path.join(BASE, 'lr_model.pkl'))
nb_model = joblib.load(os.path.join(BASE, 'nb_model.pkl'))
scaler   = joblib.load(os.path.join(BASE, 'scaler.pkl'))
with open(os.path.join(BASE, 'feature_columns.json')) as f:
    feature_columns = json.load(f)

ALL_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error  = None
    form   = {}
    if request.method == 'POST':
        try:
            form = request.form.to_dict()
            tx_type = form.get('type', 'TRANSFER')
            row = {
                'amount':         float(form['amount']),
                'oldbalanceOrg':  float(form['oldbalanceOrg']),
                'newbalanceOrig': float(form['newbalanceOrig']),
            }
            for t in ALL_TYPES:
                row[f'type_{t}'] = 1 if tx_type == t else 0
            X        = np.array([[row[c] for c in feature_columns]])
            X_scaled = scaler.transform(X)
            lr_pred  = int(lr_model.predict(X_scaled)[0])
            lr_prob  = round(float(lr_model.predict_proba(X_scaled)[0][1]) * 100, 2)
            nb_pred  = int(nb_model.predict(X_scaled)[0])
            nb_prob  = round(float(nb_model.predict_proba(X_scaled)[0][1]) * 100, 2)
            result = {
                'lr': {'label': 'FRAUD' if lr_pred == 1 else 'NORMAL', 'prob': lr_prob, 'is_fraud': lr_pred == 1},
                'nb': {'label': 'FRAUD' if nb_pred == 1 else 'NORMAL', 'prob': nb_prob, 'is_fraud': nb_pred == 1},
            }
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error, form=form)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
