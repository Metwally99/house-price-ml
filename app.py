from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
from train_models import ModelA, ModelB, ModelC, ModelD, ModelE

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

# Load Logic
input_dim = 6
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))

models = {}

def load_models():
    print("Loading models...")
    # Model A
    model_a = ModelA(input_dim)
    model_a.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model_a.pth')))
    model_a.eval()
    models['A'] = model_a
    
    # Model B
    model_b = ModelB(input_dim)
    model_b.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model_b.pth')))
    model_b.eval()
    models['B'] = model_b
    
    # Model C
    model_c = ModelC(input_dim)
    model_c.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model_c.pth')))
    model_c.eval()
    models['C'] = model_c
    
    # Model D
    model_d = ModelD(input_dim)
    model_d.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model_d.pth')))
    model_d.eval()
    models['D'] = model_d
    
    # Model E
    model_e = ModelE(input_dim)
    model_e.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model_e.pth')))
    model_e.eval()
    models['E'] = model_e
    print("Models loaded.")

load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'C') # Default to C
        
        # Extract features
        features = [
            float(data['date']),
            float(data['age']),
            float(data['distance']),
            float(data['stores']),
            float(data['latitude']),
            float(data['longitude'])
        ]
        
        # Preprocess
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        model = models.get(model_type, models['C'])
        with torch.no_grad():
            prediction = model(features_tensor)
            price = prediction.item()
            
        return jsonify({
            'success': True,
            'price': price,
            'model': model_type
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
