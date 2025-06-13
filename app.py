from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Configuración ---
MODEL_PATH = "modelo_lstm.keras"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

INPUT_STEPS = 12
OUTPUT_STEPS = 6

FEATURES = ['hora_sin', 'hora_cos', 'Wx', 'Wy', 'visibilidad', 'altura_prom_nube', 'temperatura', 'humedad_r', 'qnh']
TARGETS = ['visibilidad', 'altura_prom_nube', 'temperatura', 'humedad_r', 'qnh']

# --- Cargar modelo y scalers ---
try:
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando modelo o scalers: {e}")

@app.route('/')
def index():
    return "✅ API de predicción LSTM para clima en Kiteni está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        json_data = request.get_json(force=True)
        input_data = np.array(json_data['input'])

        # Validar forma
        if input_data.shape != (1, INPUT_STEPS, len(FEATURES)):
            return jsonify({
                'error': f'Forma inválida: {input_data.shape}. Se espera (1, {INPUT_STEPS}, {len(FEATURES)}).'
            }), 400

        # Escalar entrada
        input_reshaped = input_data.reshape(-1, len(FEATURES))
        input_scaled = scaler_X.transform(input_reshaped).reshape(1, INPUT_STEPS, len(FEATURES))

        # Predecir
        pred_scaled = model.predict(input_scaled)  # (1, OUTPUT_STEPS, len(TARGETS))
        pred_scaled = pred_scaled.reshape(OUTPUT_STEPS, len(TARGETS))

        # Desescalar salida
        pred_descaled = scaler_y.inverse_transform(pred_scaled)

        # Crear respuesta JSON
        pred_dict = {var: pred_descaled[:, i].tolist() for i, var in enumerate(TARGETS)}

        return jsonify({'prediccion': pred_dict})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
