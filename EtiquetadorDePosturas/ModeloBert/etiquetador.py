import json
import os
import time

# --- CONFIGURACIÓN ---
DIRECTORIO_HILOS = './' 
ARCHIVO_RAIZ = 'tweet_raiz.json'
ARCHIVO_PROGRESO = '_progresoV2.txt' # Almacena "ID_HILO:INDICE_TWEET_A_REVISAR"
PROGRESO_DELIMITER = ':'

# --- DEFINICIÓN DE ETIQUETAS ---
ETIQUETAS_POSTURA = {
    '1': 'De acuerdo',
    '2': 'Desacuerdo',
    '3': 'Comenta',
    '4': 'Consulta',
    's': 'saltar',
    'b': 'volver',
    'q': 'salir'
}

def limpiar_pantalla():
    """Limpia la consola para mejorar la legibilidad."""
    os.system('cls' if os.name == 'nt' else 'clear')

def cargar_tweets_raiz(archivo):
    """Carga los tweets raíz en un diccionario."""
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            lista_tweets = json.load(f)
        tweets_raiz_dict = {str(tweet['tweet_id']): tweet['titulo_noticia'] for tweet in lista_tweets}
        print(f"Se cargaron {len(tweets_raiz_dict)} tweets raíz con éxito.")
        return tweets_raiz_dict
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo '{archivo}' no tiene un formato JSON válido.")
        return None

def encontrar_primer_no_etiquetado(respuestas):
    """Encuentra el índice de la primera respuesta sin etiqueta."""
    for i, respuesta in enumerate(respuestas):
        if respuesta.get('postura') is None:
            return i
    return len(respuestas)

# --- Funciones de Progreso Actualizadas ---

def guardar_progreso(nombre_carpeta, idx_respuesta):
    """Guarda el ID del hilo y el índice del *próximo* tweet a revisar."""
    # Guarda el progreso en formato: ID_HILO:INDICE
    progreso = f"{nombre_carpeta}{PROGRESO_DELIMITER}{idx_respuesta}"
    with open(ARCHIVO_PROGRESO, 'w', encoding='utf-8') as f:
        f.write(progreso)

def leer_progreso():
    """Lee el último progreso guardado: (thread_id, tweet_index)"""
    if os.path.exists(ARCHIVO_PROGRESO):
        try:
            with open(ARCHIVO_PROGRESO, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if PROGRESO_DELIMITER in content:
                    thread_id, idx_str = content.split(PROGRESO_DELIMITER)
                    # Devuelve el ID del hilo y el índice del tweet (el que toca revisar)
                    return thread_id, int(idx_str)
        except (ValueError, FileNotFoundError):
            # En caso de error de formato, se trata como si no hubiera progreso
            return None, 0
    return None, 0

def limpiar_progreso():
    """Limpia el archivo de progreso."""
    if os.path.exists(ARCHIVO_PROGRESO):
        os.remove(ARCHIVO_PROGRESO)

# --- Bucle Principal de Etiquetado Actualizado ---

def etiquetar_hilos(directorio, tweets_raiz):
    """Función principal para recorrer los hilos y pedir etiquetas."""
    carpetas_hilos = sorted([d for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d)) and d.isdigit()])
    total_carpetas = len(carpetas_hilos)
    print(f"\nSe encontraron {total_carpetas} hilos para procesar.")

    indice_inicio_hilo = 0
    idx_tweet_a_retomar = 0
    
    # --- Lectura de Progreso al Inicio ---
    ultimo_hilo_guardado, ultimo_idx_guardado = leer_progreso()
    
    # Solo preguntamos si hay un progreso válido para retomar
    if ultimo_hilo_guardado and ultimo_hilo_guardado in carpetas_hilos:
        modo = ''
        
        limpiar_pantalla()
        while modo not in ['1', '2']:
            modo = input(f"Se encontró progreso guardado. Último punto: Hilo {ultimo_hilo_guardado} (Tweet #{ultimo_idx_guardado}).\n1: Continuar desde ese punto\n2: Empezar desde cero\nOpción: ")
        
        if modo == '1':
            indice_inicio_hilo = carpetas_hilos.index(ultimo_hilo_guardado)
            idx_tweet_a_retomar = ultimo_idx_guardado
            print(f"Reanudando desde el hilo: {ultimo_hilo_guardado}, Tweet índice: {idx_tweet_a_retomar}")
        elif modo == '2':
            limpiar_progreso()
            print("Se ha borrado el progreso. Empezando desde el principio.")
            
    # --- Bucle Principal de Hilos ---
    for i in range(indice_inicio_hilo, total_carpetas):
        nombre_carpeta = carpetas_hilos[i]
        ruta_json = os.path.join(directorio, nombre_carpeta, f"{nombre_carpeta}.json")

        if not os.path.exists(ruta_json): continue
        texto_raiz = tweets_raiz.get(str(nombre_carpeta))
        if not texto_raiz: continue

        with open(ruta_json, 'r', encoding='utf-8') as f:
            try:
                datos_hilo = json.load(f)
            except json.JSONDecodeError:
                continue
        
        respuestas = datos_hilo.get('Tweets', [])
        if not respuestas: continue
        
        # Determinar el índice de inicio
        idx_respuesta = 0
        
        # Lógica de Retoma: Solo aplica al primer hilo después de seleccionar 'Continuar'
        # Usamos el índice guardado (incluso si es 0, que es el inicio)
        if i == indice_inicio_hilo and idx_tweet_a_retomar > 0:
            # Si estamos en el hilo de la retoma, usamos el índice guardado.
            idx_respuesta = idx_tweet_a_retomar
            idx_tweet_a_retomar = 0 # Reiniciamos para que solo afecte al primer hilo de la sesión reanudada
        else:
            # Si es un hilo nuevo (i > indice_inicio_hilo) o si se eligió Empezar desde cero,
            # se parte siempre desde el tweet 0 (índice 0) de este hilo,
            # cumpliendo con la solicitud de no buscar el primer no etiquetado.
            idx_respuesta = 0

        # Si todas las respuestas ya están revisadas (o el punto de retoma es el final), saltamos.
        if idx_respuesta >= len(respuestas):
             print(f"Saltando hilo {nombre_carpeta}: Todas las respuestas ya están revisadas (inicia en {idx_respuesta}).")
             continue

        consecutive_skips = 0

        while idx_respuesta < len(respuestas):
            limpiar_pantalla()
            print(f"--- Hilo {i+1}/{total_carpetas} [Respuesta {idx_respuesta + 1}/{len(respuestas)}] --- ID: {nombre_carpeta} ---\n")
            print(f"TWEET RAÍZ:\n'{texto_raiz}'\n")
            print("="*50)

            respuesta_actual = respuestas[idx_respuesta]
            etiqueta_existente = respuesta_actual.get('postura')

            print(f"\nRESPUESTA A ETIQUETAR:\n'{respuesta_actual['texto']}'\n")
            if etiqueta_existente:
                print(f"(Etiqueta actual: {etiqueta_existente})")

            opciones_str = " | ".join([f"{k}={v}" for k, v in ETIQUETAS_POSTURA.items()])
            entrada_usuario = input(f"Elige una postura ({opciones_str}): ")

            while entrada_usuario not in ETIQUETAS_POSTURA:
                entrada_usuario = input("Opción inválida. Intenta de nuevo: ")

            if entrada_usuario == 'q':
                # Al salir, guardamos el índice actual (el que no fue etiquetado)
                guardar_progreso(nombre_carpeta, idx_respuesta)
                print("Saliendo y guardando progreso...")
                return
            
            if entrada_usuario == 'b':
                consecutive_skips = 0
                idx_respuesta = max(0, idx_respuesta - 1)
                continue
            
            if entrada_usuario == 's':
                consecutive_skips += 1
                if consecutive_skips >= 4:
                    saltar_hilo = input("Has saltado 4 veces seguidas. ¿Quieres saltar al siguiente hilo? (s/n): ")
                    if saltar_hilo.lower() == 's':
                        # Guardar progreso del siguiente tweet antes de romper el hilo
                        guardar_progreso(nombre_carpeta, idx_respuesta + 1)
                        print("Saltando al siguiente hilo...")
                        time.sleep(1)
                        break
                    else:
                        consecutive_skips = 0
                idx_respuesta += 1
                
                # Si se salta, guardar el progreso del siguiente tweet a revisar
                if idx_respuesta < len(respuestas):
                     guardar_progreso(nombre_carpeta, idx_respuesta)
                continue

            # Si se etiqueta, se resetea el contador de saltos
            consecutive_skips = 0
            respuesta_actual['postura'] = ETIQUETAS_POSTURA[entrada_usuario]

            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(datos_hilo, f, ensure_ascii=False, indent=2)
            
            print(f"Etiquetado como: {respuesta_actual['postura']} (Progreso guardado)")
            time.sleep(0.5)
            idx_respuesta += 1

            # Después de etiquetar con éxito, guardamos el progreso para el *próximo* tweet.
            if idx_respuesta < len(respuestas):
                guardar_progreso(nombre_carpeta, idx_respuesta)
        
        if idx_respuesta >= len(respuestas):
              print(f"\nTodas las respuestas del hilo {nombre_carpeta} han sido revisadas.")
              input("Presiona Enter para continuar con el siguiente hilo.")
              # Si se completa el hilo, no se necesita guardar progreso aquí,
              # el bucle 'for' pasará al siguiente y si se sale, el siguiente hilo
              # se retomará desde el inicio o el primer no etiquetado.

    limpiar_progreso()
    print("\nSe han etiquetado todos los hilos.")

if __name__ == "__main__":
    tweets_raiz_diccionario = cargar_tweets_raiz(ARCHIVO_RAIZ)
    if tweets_raiz_diccionario:
        etiquetar_hilos(DIRECTORIO_HILOS, tweets_raiz_diccionario)
    print("\nProceso de etiquetado finalizado.")
