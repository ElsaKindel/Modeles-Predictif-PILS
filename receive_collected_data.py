import requests
import json

# API URL
API_URL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"

# Access token
# Lire le jeton depuis le fichier jeton.txt
with open("jeton.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()


def request_data_from_api(restaurant, date, json_name, day=None, lookback=4):
    """date au format :"2024-12-03T12:00:00"
    """
    # Data requested
    donnees_reelles = { 
        "action": "getWaitingTime",
        "restaurant": restaurant,
        "date": date+"T12:00:00",
        "token" : ACCESS_TOKEN,
    }

    try:
        response = requests.post(API_URL, data=json.dumps(donnees_reelles)) #, params=param
        
        # Verify answer
        if response.status_code == 200:
            print("Données reçues avec succès !")
            #print("Réponse de l'API :", response.json())
        else:
            print(f"Échec de la requête : {response.status_code}")
            print("Détails :", response.text)
    except requests.exceptions.RequestException as e:
        print("Erreur lors de la communication avec l'API :", e)

    # Sauvegarder dans un fichier
    file_address = "Fichiers_JSON_donnees_reelles/"+json_name
    with open(file_address, "a") as f:
        #print("Données à enregistrer :", response.json())
        json.dump(response.json(), f)

    print("Données enregistrée dans "+json_name)
    return response.json()

if __name__ == "__main__":
    request_data_from_api(restaurant="olivier", date = "2024-12-03T01:00:00", day=None, lookback=4)
