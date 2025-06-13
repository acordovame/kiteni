from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Configuración del modelo ---
MODEL_PATH = "modelo_lstm.keras"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
INPUT_STEPS = 12   # pasos de entrada (ventana deslizante)
OUTPUT_STEPS = 12  # pasos a predecir (12 predicciones futuras)

# Features utilizadas (deben coincidir con entrenamiento)
FEATURES = ['day_sin', 'day_cos', 'hora_sin', 'hora_cos', 'Wx', 'Wy', 'visibilidad',
            'altura_prom_nube', 'temperatura', 'qnh', 'humedad_r']

# Variables objetivo que predice el modelo
TARGETS = ['visibilidad', 'altura_prom_nube', 'temperatura', 'qnh', 'humedad_r']

# --- Cargar modelo y escaladores ---
try:
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Error cargando modelo o escaladores: {e}")

@app.route('/')
def index():
    return "✅ API de predicción LSTM multivariada para clima en Kiteni está activa."

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        json_data = request.get_json(force=True)

        # Obtener y validar entrada
        input_data = np.array(json_data['input'])

        if input_data.shape != (1, INPUT_STEPS, len(FEATURES)):
            return jsonify({
                'error': f'Forma de entrada inválida: {input_data.shape}. Se esperaba (1, {INPUT_STEPS}, {len(FEATURES)}).'
            }), 400

        # Escalar entrada
        input_reshaped = input_data.reshape(-1, len(FEATURES))
        input_scaled = scaler_X.transform(input_reshaped).reshape(1, INPUT_STEPS, len(FEATURES))

        # Predicción
        pred_scaled = model.predict(input_scaled)  # → (1, OUTPUT_STEPS, len(TARGETS))
        pred_scaled = pred_scaled.reshape(OUTPUT_STEPS, len(TARGETS))

        # Desescalar salida
        pred_descaled = scaler_y.inverse_transform(pred_scaled)

        # Construcción de respuesta
        pred_dict = {
            var: pred_descaled[:, i].tolist() for i, var in enumerate(TARGETS)
        }

        return jsonify({'prediccion': pred_dict})

    except Exception as e:
        return jsonify({'error': f'❌ Error en predicción: {str(e)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
