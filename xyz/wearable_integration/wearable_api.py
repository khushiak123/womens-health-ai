# python
import requests
import google_fitbit_api_client

client = google_fitbit_api_client.GoogleFitAPI(client_id="YOUR_CLIENT_ID", client_secret="YOUR_CLIENT_SECRET")
health_data = client.get_daily_activity()
response = requests.post("http://127.0.0.1:5000/predict", json=health_data)
print("AI Health Prediction:", response.json())