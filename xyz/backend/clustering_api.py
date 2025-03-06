# python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

app = Flask(__name__)
CORS(app)

@app.route("/segment", methods=["POST"])
def segment():
    # Expecting JSON data with health attributes (e.g. Age, BMI, Conditions, etc.)
    data = request.json
    df = pd.DataFrame([data])
    
    # Standardize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Hierarchical Clustering segmentation
    hierarchical_model = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical_model.fit_predict(df_scaled)
    
    # Gaussian Mixture Model segmentation
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(df_scaled)
    
    recommendations = {
        0: "Focus on preventive health and regular screening.",
        1: "Specialized support for reproductive health and wellness.",
        2: "Comprehensive care for chronic condition management."
    }
    
    response = {
        "hierarchical_cluster": int(hierarchical_labels[0]),
        "gmm_cluster": int(gmm_labels[0]),
        "recommendation": recommendations.get(int(gmm_labels[0]), "No recommendation available.")
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5001)