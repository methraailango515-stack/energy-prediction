from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["*"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features.pkl')

model = joblib.load(MODEL_PATH)
features_list = joblib.load(FEATURES_PATH)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            hour INTEGER,
            day_of_week INTEGER,
            num_appliances INTEGER,
            is_weekend INTEGER,
            prediction REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


@app.route('/')
def home():
    return jsonify({"message": "Energy Prediction API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        month = int(data.get('month', datetime.now().month))

        row = {
            'hour': int(data['hour']),
            'day_of_week': int(data['day_of_week']),
            'is_weekend': int(data['is_weekend']),
            'month': month,
            'Voltage': float(data.get('voltage', 240.0)),
            'Global_intensity': float(data.get('current', 5.0)),
        }
        features = np.array([[row[f] for f in features_list]])
        prediction = float(model.predict(features)[0])

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO history (temperature, hour, day_of_week, num_appliances, is_weekend, prediction, created_at) VALUES (?,?,?,?,?,?,?)',
            (data.get('temperature', 0),
             data['hour'],
             data['day_of_week'],
             data.get('num_appliances', 0),
             data['is_weekend'],
             round(prediction, 2),
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "unit": "kWh"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM history ORDER BY id DESC LIMIT 50').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    try:
        importances = model.feature_importances_.tolist()
        result = [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in zip(features_list, importances)
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)