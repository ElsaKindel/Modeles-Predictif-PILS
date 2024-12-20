import requests
import json

# API URL
API_URL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"

# Access token
# Lire le jeton depuis le fichier jeton.txt
with open("jeton.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()

# Data to send (test)
predictions = { # False prediction, for testing only
    "action": "updatePredictedTime",
    "predictionArray": [2,3,6,11,13,18,26,22,27,24,11,9,6], 
    "restaurant": "ri lunch",
    "date":"2025-01-14T12:00:00",
    "token" : ACCESS_TOKEN,
}

# Filling if the headers
headers = {
    "Content-Type": "application/json",
}

# POST request
try:
    response = requests.post(API_URL, headers=headers, data=json.dumps(predictions))
    
    # Verify answer
    if response.json()["status"] == "error":
        print(f"Échec de l'envoi : {response.status_code}")
        print("Détails :", response.text)
    elif response.status_code == 200:
        print("Prédictions envoyées avec succès !")
        print("Réponse de l'API :", response.json())
    else:
        print(f"Échec de l'envoi : {response.status_code}")
        print("Détails :", response.text)
except requests.exceptions.RequestException as e:
    print("Erreur lors de la communication avec l'API :", e)