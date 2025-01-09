import requests
import json

# API URL
API_URL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"

# Access token
# Lire le jeton depuis le fichier jeton.txt
with open("jeton.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()

# GET request
try:
    response = requests.get(API_URL)
    #print(response.text)
    
    # Verify answer
    if response.status_code == 200:
        print("Données reçues avec succès !")
        print("Réponse de l'API :", response.json())
    else:
        print(f"Échec de la requête : {response.status_code}")
        print("Détails :", response.text)
except requests.exceptions.RequestException as e:
    print("Erreur lors de la communication avec l'API :", e)

