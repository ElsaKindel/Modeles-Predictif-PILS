from datetime import date
from prediction import calculate_prediction, save_as_json, send_data_to_API

today = date.today()
restaurant = "olivier"
real_data_json = "ref_data_for"+str(today)+"_"+restaurant


# Teste calculate_prediction
prediction = calculate_prediction(restaurant, real_data_json, today, lookback=4)

# Enregister le fichier json des prédictions
json_name = save_as_json(prediction, today, restaurant)

#Envoyer les données à l'API
send_data_to_API(json_name)

