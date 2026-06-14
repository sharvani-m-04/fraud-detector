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
TYPE_MAP  = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}

TUNING_RESULTS = {
    'lr':  {'params': 'C=10, solver=lbfgs',                              'f1': '0.885'},
    'nb':  {'params': 'var_smoothing=1e-9',                               'f1': '0.501'},
    'rf':  {'params': 'max_depth=10, n_estimators=100',                   'f1': '0.992'},
    'xgb': {'params': 'learning_rate=0.3, max_depth=5, n_estimators=100', 'f1': '0.993'},
    'svm': {'params': 'C=1, kernel=rbf',                                  'f1': '0.884'},
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
            amount      = float(form['amount'])
            old_balance = float(form['oldbalanceOrg'])
            new_balance = float(form['newbalanceOrig'])

            # 8-feature input for LR and NB (one-hot encoded)
            row = {'amount': amount, 'oldbalanceOrg': old_balance, 'newbalanceOrig': new_balance}
            for t in ALL_TYPES:
                row[f'type_{t}'] = 1 if tx_type == t else 0
            X8        = np.array([[row[c] for c in feature_columns]])
            X8_scaled = scaler.transform(X8)

            # 4-feature input for RF, XGB, SVM (label encoded)
            type_encoded = TYPE_MAP.get(tx_type, 4)
            X4        = np.array([[type_encoded, amount, old_balance, new_balance]])
            X4_scaled = scaler.transform(X8)[:, :4]

            def predict(model, X):
                pred = int(model.predict(X)[0])
                prob = round(float(model.predict_proba(X)[0][1]) * 100, 2)
                return {'label': 'FRAUD' if pred == 1 else 'NORMAL', 'prob': prob, 'is_fraud': pred == 1}

            result = {
                'lr':  predict(lr_model,  X8_scaled),
                'nb':  predict(nb_model,  X8_scaled),
                'rf':  predict(rf_model,  X4_scaled),
                'xgb': predict(xgb_model, X4_scaled),
                'svm': predict(svm_model, X4_scaled),
            }
        except Exception as e:
            error = str(e)
    return render_template('index.html', result=result, error=error, form=form, tuning=TUNING_RESULTS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
