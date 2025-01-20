import numpy as np
import os.path
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from receive_collected_data import request_data_from_api
from datetime import timedelta, date
from send_predictions import send_data_to_API

MONDAY = "monday"
TUESDAY = "tuesday"
WEDNESDAY = "wednesday"
THURSDAY = "thursday"
FRIDAY = "friday"
SATURDAY = "saturday"
SUNDAY = "sunday"

def calculate_days_to_request(prediction_date, lookback=4):
    """Avec date au format 2024-12-03"""
    day = date.fromisoformat(prediction_date)
    time_delta = timedelta(days=7)
    days_to_request = []
    for i in range(lookback):
        day = day - time_delta
        days_to_request.append(day)
    return days_to_request #array fo dates with shape :"2024-12-03"

def get_data(restaurant, requested_days, json_name, lookback_weeks):
    """Effectue des requêtes à l'API pour obtenir les informations voulues.
    Retourne les données au format np.ndarray(lookback_week*120) -> à sous-échantillonner"""

    if os.path.isfile("Fichiers_JSON_donnees_reelles/"+json_name):
        raise RuntimeError("JSON file already exists")
    
    collected_data = []
    downsampling_factor = 10 #On nous envoie 120 donnée, on en veut 12

    for day in requested_days:
        daily_data = request_data_from_api(restaurant= restaurant,
                                           date = day.isoformat(), 
                                           json_name = json_name, 
                                           lookback=lookback_weeks)
        daily_data = daily_data["array"][1:][::downsampling_factor] 
        #on commence à 1 pour virer la date
        #print(daily_data)
        collected_data.append(daily_data) 
    collected_data = np.array(collected_data) #turns nested list into ndarray
    return collected_data

def prepare_lstm_data_weekly_pattern(data, lookback_weeks=4):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data)
    
    # Vérification des formes avant de retourner
    print(f"Longueur de X : {len(scaled_values)}")
    if len(scaled_values) == 0 :
        raise ValueError("X ou y est vide. Vérifiez les paramètres de lookback_weeks ou les données d'entrée.")

    return scaled_values, scaler

def pred_one_day(X, model, scaler):
    X = np.expand_dims(X, axis=0) #ajoute la batch size, obligatoire en LSTM
    y_pred = model.predict(X)
    y_pred_rescaled = scaler.inverse_transform(y_pred)  # Revenir aux valeurs originales
    #print("Prédictions :", y_pred_rescaled)

    return y_pred_rescaled

def pred_one_week(data, model):
    """scaler = MinMaxScaler()
    X_test_one_day = X[22] #jour random pris dans l'historique généré
    # X[22] est de forme 7*24 puisqu'il contient les données des 7 jours précédents (lookback = 7) 
    # et qu'il y a 24 échantillons pour chaque service
    X_test_one_day = np.expand_dims(X_test_one_day, axis=0) #ajoute la batch size, obligatoire en LSTM
    y_pred = model.predict(X_test_one_day)
    y_pred_rescaled = scaler.inverse_transform(y_pred)  # Revenir aux valeurs originales
    print("Prédictions :", y_pred_rescaled)"""

    return

def calculate_prediction(restaurant, json_name, day=None, lookback =4):
    """Paramètres: 
    - day : date au format iso ou None si on veut prédire tous les jours de la semaine qui suit
    - lookback : 4 par défaut, nombre de semaines précédentes prises en compte dans la prédiction"""

    # Charger le modèle
    model = load_model("modele_lstm_attente.keras")
    print("Modèle LSTM chargé avec succès.")

    # Lister les jours sur lesquels baser la prédiction
    days_to_request = calculate_days_to_request(day)

    # Obtenir les données des semaines précédentes
    lookback_weeks=4 #Nombre de semaines sur lesquelles se base le modèle pour faire sa prédiction
    data = get_data(restaurant= restaurant, 
                    requested_days=days_to_request, 
                    json_name=json_name,  
                    lookback_weeks=lookback)

    # Préparer les données
    data, scaler = prepare_lstm_data_weekly_pattern(data, lookback_weeks=4)

    # Prédiction
    prediction = pred_one_day(data, model, scaler)

    # Affichage prédiction
    if isinstance(day, str):
        text_day = day
    elif day is None:
        text_day = "all week"
    else:
        raise ValueError("Not valid day value. Select a day of the week or None to predict the all week")
   
    print("Prediction for", text_day, "with ", lookback, "weeks of lookback : ", prediction)
    return prediction

def save_as_json(data, day, restaurant):
    # with restaurant among : olivier, Ri Déjeuner and Ri Diner
     # Sauvegarder dans un fichier
    json_name = day+"_"+restaurant
    file_address = "Fichiers_JSON_predictions/"+json_name
    json_data = {"prediction":data.tolist()[0], "restaurant": restaurant, "date":day}
    with open(file_address, "w") as f:
        #print("Données à enregistrer :", response.json())
        json.dump(json_data, f)

    print("Données enregistrée dans "+ file_address)
    return json_name

if __name__=="__main__":

    today = "2025-01-09"
    fichier_test = "test.json"
    restaurant = "olivier"
    os.remove("Fichiers_JSON_donnees_reelles/"+fichier_test)

    # Teste calculate_days_to_request
    """days_to_request = calculate_days_to_request(today) #Donne la liste des jours qui serviront à faire la prédiction de ce jour-ci.
    print("Jours à requérir à l'API pour effectuer la prédiction :")
    for i in range(len(days_to_request)):
        print(days_to_request[i])"""
    
    # Teste get_data
    """    data = get_data("olivier", days_to_request, fichier_test, 4)
    print("Données à nourir au modèle: ", data)"""
    
    # Teste calculate_prediction
    prediction = calculate_prediction(restaurant, fichier_test, today, lookback=4)

    # Enregister le fichier json des prédictions
    json_name = save_as_json(prediction, today, restaurant)

    #Envoyer les données à l'API
    send_data_to_API(json_name)
    