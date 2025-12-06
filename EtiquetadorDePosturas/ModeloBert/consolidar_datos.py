import os
import json
import pandas as pd

# --- CONFIGURACIÓN ---
DIRECTORIO_HILOS = './'
ARCHIVO_SALIDA_CSV = 'datos_etiquetados.csv'

def consolidar_etiquetas(directorio, archivo_salida):
    """
    Recorre los archivos JSON, extrae los textos con su postura etiquetada
    y los guarda en un único archivo CSV.
    """
    datos_etiquetados = []
    
    print("Iniciando la consolidación de datos etiquetados...")
    
    # Obtener todas las carpetas de hilos
    carpetas_hilos = [d for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d)) and d.isdigit()]

    if not carpetas_hilos:
        print("❌ No se encontraron carpetas de hilos en el directorio.")
        return

    # Recorrer cada carpeta
    for nombre_carpeta in carpetas_hilos:
        ruta_json = os.path.join(directorio, nombre_carpeta, f"{nombre_carpeta}.json")
        
        if not os.path.exists(ruta_json):
            continue
            
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos_hilo = json.load(f)
            
        # Buscar respuestas que ya tengan una etiqueta
        for respuesta in datos_hilo.get('Tweets', []):
            if respuesta.get('postura') and respuesta['postura'] is not None:
                datos_etiquetados.append({
                    'texto': respuesta['texto'],
                    'postura': respuesta['postura']
                })

    if not datos_etiquetados:
        print("⚠️ No se encontraron datos etiquetados. Asegúrate de haber ejecutado primero 'etiquetador.py'.")
        return

    # Convertir la lista de datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_etiquetados)
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ ¡Éxito! Se consolidaron {len(df)} registros.")
    print(f"Los datos etiquetados se han guardado en: '{archivo_salida}'")
    print("\nDistribución de clases:")
    print(df['postura'].value_counts())


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    consolidar_etiquetas(DIRECTORIO_HILOS, ARCHIVO_SALIDA_CSV)