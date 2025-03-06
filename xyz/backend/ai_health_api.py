import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from groq import Groq

app = Flask(__name__)
CORS(app)

# Initialize Groq client with your API key
client = Groq(api_key="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")

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

def get_ai_insights(health_data):
    prompt = f"""
    Analyze the following health metrics and provide detailed recommendations:
    Age: {health_data['Age']}
    BMI: {health_data['BMI']}
    Chronic Conditions: {health_data['Chronic_Conditions']}
    Physical Activity Level: {health_data['Physical_Activity']}
    Mental Health Score: {health_data['Mental_Health_Score']}
    Reproductive Health Score: {health_data['Reproductive_Health_Score']}
    Menopause Status: {'Yes' if health_data['Menopause_Status'] == 1 else 'No'}
    
    Provide specific recommendations for:
    1. Lifestyle changes
    2. Health monitoring
    3. Preventive measures
    4. Risk factors to watch
    """
    
    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a healthcare AI assistant providing personalized health recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

# Initialize model and load weights
autoencoder = Autoencoder(7)
model_path = os.path.join(os.path.dirname(__file__), "autoencoder_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Run train_autoencoder.py first.")

autoencoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
autoencoder.eval()

# Initialize scaler
scaler = StandardScaler()
scaler.fit([[30, 25, 2, 5, 70, 50, 1]])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)
        
        X_tensor = torch.FloatTensor(df_scaled)
        with torch.no_grad():
            encoded_features = autoencoder.encoder(X_tensor).numpy()

        sample_buffer = np.vstack([
            encoded_features,
            np.random.normal(encoded_features.mean(), encoded_features.std(), (3, encoded_features.shape[1]))
        ])

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = kmeans.fit_predict(sample_buffer)
        
        ai_insights = get_ai_insights(data)

        recommendations = {
            0: "Exercise regularly, manage stress, focus on mental health.",
            1: "Comprehensive care including chronic disease management and specialized support."
        }

        response = {
            "cluster": int(kmeans_labels[0]),
            "base_recommendation": recommendations[int(kmeans_labels[0])],
            "ai_insights": ai_insights,
            "risk_score": float(encoded_features.mean()),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting AI Health API on port 5000...")
    app.run(debug=True, port=5000)