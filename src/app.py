from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
import pandas as pd
import threading, time, random
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

import shap

# ==================== Flask + SocketIO ====================
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ==================== MongoDB ====================
client = MongoClient("mongodb://localhost:27017/")
db = client["industrial_twin_db"]
collection = db["predicted_readings"]

# ==================== Load Dataset ====================
print("📘 Loading dataset & training Hybrid AI model...")
df = pd.read_csv("src/ai4i2020.csv")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

df_majority = df[df["Machine failure"] == 0]
df_minority = df[df["Machine failure"] == 1]

df_balanced = pd.concat([
    df_majority,
    resample(df_minority, replace=True,
             n_samples=len(df_majority), random_state=42)
])

X = df_balanced[FEATURES]
y = df_balanced["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_scaled, y)

explainer = shap.TreeExplainer(rf_model)

print("✅ Hybrid Model + Explainability Ready")

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/readings")
def readings():
    data = list(collection.find({}, {"_id": 0}).sort("_id", -1).limit(50))
    return render_template("readings.html", data=data)

@app.route("/api/readings")
def api_readings():
    data = list(collection.find({}, {"_id": 0}).sort("_id", -1).limit(50))
    return jsonify(data)

# ==================== Background Simulation ====================
thread_started = False
thread_lock = threading.Lock()

def simulate_data():
    print("🧠 Simulation thread started")
    while True:
        try:
            sample = df_balanced.sample(1).copy()
            sample_df = sample[FEATURES].copy()

            sample_df.iloc[:, 0] += random.uniform(-5, 5)
            sample_df.iloc[:, 1] += random.uniform(-5, 5)
            sample_df.iloc[:, 3] += random.uniform(-10, 10)

            scaled_sample = scaler.transform(sample_df)

            rf_p = float(rf_model.predict_proba(scaled_sample)[0][1])
            svm_p = float(svm_model.predict_proba(scaled_sample)[0][1])
            final_p = 0.6 * rf_p + 0.4 * svm_p

            hazard = "NORMAL" if final_p < 0.3 else "MEDIUM" if final_p < 0.6 else "CRITICAL"

            # ==================== SHAP (FIXED) ====================
            shap_values = explainer.shap_values(scaled_sample)

            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_vals = shap_values[1][0]   # failure class
                else:
                    shap_vals = shap_values[0][0]   # fallback
            elif hasattr(shap_values, "values"):
                shap_vals = shap_values.values[0]
            else:
                shap_vals = shap_values[0]

            shap_vals = np.ravel(shap_vals).astype(float)
            # =====================================================

            explanation = sorted(
                zip(FEATURES, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]

            reading = {
                "temp": round(float(sample_df.iloc[0, 0]), 2),
                "pressure": round(float(sample_df.iloc[0, 1]), 2),
                "vibration": round(float(sample_df.iloc[0, 3]), 2),
            }

            socketio.emit("sensor_reading", {
                "reading": reading,
                "hazard_level": hazard,
                "explanation": explanation
            })

            collection.insert_one({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature_K": reading["temp"],
                "pressure_bar": reading["pressure"],
                "vibration_mms": reading["vibration"],
                "hazard_level": hazard
            })

            time.sleep(18)

        except Exception as e:
            print("⚠️ Simulation error:", e)
            time.sleep(2)

@socketio.on("connect")
def on_connect():
    global thread_started
    with thread_lock:
        if not thread_started:
            socketio.start_background_task(simulate_data)
            thread_started = True
            print("🌐 Client connected — simulation running")

# ==================== Run ====================
if __name__ == "__main__":
    print("🚀 Running at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
