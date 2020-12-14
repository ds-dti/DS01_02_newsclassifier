import requests

myText = "{\"text\":\"Menteri Ketenagakerjaan Ida Fauziyah mengatakan, untuk melindungi   Pekerja  Migran Indonesia (PMI), kuncinya adalah sinergitas dan kolaborasi seluruh pihak.\"}"
url = "http://127.0.0.1:8000/predict/"
response = requests.post(url, data=myText)
print(response.json())