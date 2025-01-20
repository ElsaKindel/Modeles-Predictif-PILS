import requests
import json

# API URL
API_URL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"

# Access token
# Lire le jeton depuis le fichier jeton.txt
with open("jeton.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()

def send_data_to_API(json_name):

    # Charger les prédictions depuis le fichier
    file_name = "Fichiers_JSON_predictions/"+json_name
    with open(file_name, "r") as f:
        info = json.load(f)

    # Data to send
    predictions = { # False prediction, for testing only
        "action": "updatePredictedTime",
        "predictionArray": info["prediction"],
        "restaurant": info["restaurant"],
        "date":info["date"],
        "token" : ACCESS_TOKEN,
    }

    # Filling the headers
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