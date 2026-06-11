from flask import Flask, request, render_template
import joblib, json, numpy as np, os

app = Flask(__name__, template_folder='../frontend/templates')

BASE = os.path.dirname(os.path.abspath(__file__))
lr_model  = joblib.load(os.path.join(BASE, 'lr_model.pkl'))
nb_model  = joblib.load(os.path.join(BASE, 'nb_model.pkl'))
rf_model  = joblib.load(os.path.join(BASE, 'rf_model.pkl'))
xgb_model = joblib.load(os.path.join(BASE, 'xgb_model.pkl'))
svm_model = joblib.load(os.path.join(BASE, 'svm_model.pkl'))
scaler    = joblib.load(os.path.join(BASE, 'scaler.pkl'))
with open(os.path.join(BASE, 'feature_columns.json')) as f:
    feature_columns = json.load(f)

ALL_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

TUNING_RESULTS = {
    'lr':  {'params': 'C=10, solver=lbfgs',                          'f1': '0.885'},
    'nb':  {'params': 'var_smoothing=1e-9',                           'f1': '0.501'},
    'rf':  {'params': 'max_depth=10, n_estimators=100',               'f1': '0.992'},
    'xgb': {'params': 'learning_rate=0.3, max_depth=5, n_estimators=100', 'f1': '0.993'},
    'svm': {'params': 'C=1, kernel=rbf',                              'f1': '0.884'},
}

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

            def predict(model):
                pred = int(model.predict(X_scaled)[0])
                prob = round(float(model.predict_proba(X_scaled)[0][1]) * 100, 2)
                return {'label': 'FRAUD' if pred == 1 else 'NORMAL', 'prob': prob, 'is_fraud': pred == 1}

            result = {
                'lr':  predict(lr_model),
                'nb':  predict(nb_model),
                'rf':  predict(rf_model),
                'xgb': predict(xgb_model),
                'svm': predict(svm_model),
            }
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error, form=form, tuning=TUNING_RESULTS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
