import requests
import json

# API URL
API_URL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"

# Access token
# Lire le jeton depuis le fichier jeton.txt
with open("jeton.txt", "r") as file:
    ACCESS_TOKEN = file.read().strip()