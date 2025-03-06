import json
import requests
import os 
from pathlib import Path


class Ingestiones():
    def __init__(self):
        self.ruta_static=os.path.join("src","pad20251","static")
        self.ruta_actual= str(patch.cwd())
        self.ruta_static="{}/src/pad20251/static/".format(self.ruta_actual)
        directorio= os.path.dirname(self.ruta_actual)
       
        if not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True) 
        
    def leer_api(self,url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"error al obtener datos de la API: {e}")
            return None
 
    def escribir_json(self,datos,nombre_archivo="datos.json"):
        if datos is None:
            print("No hay datos para guardar.")
            return 
        ruta_completa = os.path.join(self.ruta_static, nombre_archivo) 
        try:
            with open(ruta_completa, "w", encoding="utf-8") as archivo:
                json.dump(datos, archivo, indent=4, ensure_ascii=False)
            print(f"Datos guardados en {ruta_completa}")
        except Exception as e:
            print("Error al escribir el archivo JSON: {e}") 
        
# creacion de la instancia 
ingestion = Ingestiones ()
datos_json = ingestion.leer_api("https://hp-api.onrender.com/api/characters") 

# imprimir información 
print ("esta es la ruta estatica:", ingestion.ruta_static) 
if datos_json:
    print("Datos obtenidos correctamente.")
    ingestion.escribir_json(datos_json)

else:
     print("Error: No se pudieron obtener datos de la API.")


