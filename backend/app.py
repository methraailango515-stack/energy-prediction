from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, sqlite3, os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["*"])

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, 'model.pkl')
FEATURES_PATH = os.path.join(BASE, 'features.pkl')
DB_PATH = os.path.join(BASE, 'predictions.db')

def train_if_missing():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        return
    from sklearn.ensemble import RandomForestRegressor
    np.random.seed(42); n = 5000
    hour = np.random.randint(0,24,n); dow = np.random.randint(0,7,n)
    we = (dow>=5).astype(int); mon = np.random.randint(1,13,n)
    volt = np.random.uniform(220,250,n); cur = np.random.uniform(0.2,20,n)
    y = volt*cur*0.9 + hour*0.5 + np.random.normal(0,3,n)
    X = np.column_stack([hour,dow,we,mon,volt,cur])
    feats = ['hour','day_of_week','is_weekend','month','Voltage','Global_intensity']
    m = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    m.fit(X,y)
    joblib.dump(m, MODEL_PATH)
    joblib.dump(feats, FEATURES_PATH)
    print("Model auto-trained.")

train_if_missing()
model = joblib.load(MODEL_PATH)
features_list = joblib.load(FEATURES_PATH)

def init_db():
    c = sqlite3.connect(DB_PATH)
    c.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, temperature REAL, hour INTEGER, day_of_week INTEGER, num_appliances INTEGER, is_weekend INTEGER, prediction REAL, created_at TEXT)''')
    c.commit(); c.close()
init_db()

@app.route('/')
def home():
    return jsonify({"message":"Energy Prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()
        month = int(d.get('month', datetime.now().month))
        row = {'hour':int(d['hour']),'day_of_week':int(d['day_of_week']),
               'is_weekend':int(d['is_weekend']),'month':month,
               'Voltage':float(d.get('voltage',240.0)),
               'Global_intensity':float(d.get('current',5.0))}
        X = np.array([[row[f] for f in features_list]])
        p = float(model.predict(X)[0])
        c = sqlite3.connect(DB_PATH)
        c.execute('INSERT INTO history (temperature,hour,day_of_week,num_appliances,is_weekend,prediction,created_at) VALUES (?,?,?,?,?,?,?)',
                  (d.get('temperature',0),d['hour'],d['day_of_week'],d.get('num_appliances',0),d['is_weekend'],round(p,2),datetime.now().isoformat()))
        c.commit(); c.close()
        return jsonify({"success":True,"prediction":round(p,2),"unit":"kWh"})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400

@app.route('/history', methods=['GET'])
def history():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    rows = c.execute('SELECT * FROM history ORDER BY id DESC LIMIT 50').fetchall(); c.close()
    return jsonify([dict(r) for r in rows])

@app.route('/feature-importance', methods=['GET'])
def feat():
    try:
        imp = model.feature_importances_.tolist()
        r = [{"feature":n,"importance":round(i,4)} for n,i in zip(features_list,imp)]
        r.sort(key=lambda x: x["importance"], reverse=True)
        return jsonify(r)
    except Exception as e:
        return jsonify({"error":str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
