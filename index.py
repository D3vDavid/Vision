import whisper
import pyaudio
import numpy as np
import time
from ultralytics import YOLO
import cv2
import threading
from gtts import gTTS
import os
import pygame  # Usamos pygame para reproducir el audio

# Cargar el modelo de Whisper
model = whisper.load_model("base")

# Cargar el modelo de YOLOv8 (con verbose=False para desactivar logs)
yolo_model = YOLO('epoch115.pt', verbose=False)  # Asegúrate de tener el modelo yolov8 (puede ser 'yolov8n.pt', 'yolov8s.pt', etc.)

# Variable global para los objetos detectados
objetos_detectados = []
audio_data_buffer = []
mensaje_para_tts = ""

# Función de grabación de audio
def grabar_audio():
    global audio_data_buffer
    # Configuración del micrófono
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    
    print("Escuchando...")

    while True:
        data = stream.read(1024)
        audio_data_buffer.append(data)  # Guardamos el audio en un buffer

# Función para reconocer audio
def reconocer_audio():
    global audio_data_buffer
    while True:
        if len(audio_data_buffer) > 0:
            audio_data = b''.join(audio_data_buffer)  # Unimos el audio en el buffer
            audio_data_buffer = []  # Limpiar el buffer después de tomar el audio
            
            # Convertir el audio a un formato de tipo float32
            audio_np = np.frombuffer(audio_data, dtype=np.int16)  # Leer el buffer de audio como int16
            audio_float = audio_np.astype(np.float32)  # Convertir a float32

            # Normalizar el rango de valores de -1 a 1
            audio_float /= np.max(np.abs(audio_float))  # Normaliza el audio

            # Transcribir el audio usando Whisper
            result = model.transcribe(audio_float, language='es')
            print(f"He escuchado: {result['text']}")

            return result['text']

# Función de detección de objetos usando YOLOv8
def detectar_objetos(imagen):
    # Realiza la detección de objetos en la imagen
    results = yolo_model(imagen)  # Esto devuelve una lista de resultados, no un objeto 'names'

    # Extraer las predicciones de la primera (y única) imagen procesada
    objetos_detectados_local = results[0].names  # Extraemos los nombres de los objetos detectados
    boxes = results[0].boxes  # Coordenadas de las cajas de detección

    # Cambiar el nombre del objeto con índice 0 a "cuchillo"
    if len(objetos_detectados_local) > 0:
        objetos_detectados_local[0] = "cuchillo"  # Asignar "cuchillo" al objeto con índice 0

    # Convertir los índices de las clases detectadas a sus nombres
    nombres_objetos = [objetos_detectados_local[int(idx)] for idx in results[0].boxes.cls]

    return nombres_objetos, boxes

# Función para escuchar y responder, ejecutada en un hilo separado
def escuchar_y_responder():
    global objetos_detectados

    cap = cv2.VideoCapture(0)  # Captura desde la cámara
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar la imagen")
            break

        # Detectar objetos en la imagen capturada
        objetos_detectados, boxes = detectar_objetos(frame)

        # Dibujar las cajas de detección y etiquetas
        for i, box in enumerate(boxes):
            # Obtener las coordenadas de la caja de la predicción
            x1, y1, x2, y2 = map(int, box.xywh[0])  # Coordenadas de la caja delimitadora

            # Dibujar la caja alrededor del objeto detectado
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Etiqueta con el nombre del objeto
            label = objetos_detectados[i]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar la imagen con las detecciones
        cv2.imshow("Vista en tiempo real de YOLOv8", frame)

        # Verificar si se presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

# Función para generar nombres únicos para los archivos TTS
def generar_nombre_unico():
    """Genera un nombre único basado en el timestamp o un contador"""
    timestamp = int(time.time())  # Usa el timestamp para generar nombres únicos
    return f"mensaje_{timestamp}.mp3"

# Función para convertir texto a voz y reproducirlo desde un archivo
def decir_mensaje(mensaje):
    # Verifica si el sistema de pygame está inicializado
    if not pygame.mixer.get_init():
        pygame.mixer.init()  # Inicializar el mezclador de pygame si no está

    # Generar un nombre único para el archivo de audio
    mensaje_archivo = generar_nombre_unico()

    # Crear el archivo de audio con gTTS
    tts = gTTS(text=mensaje, lang='es')
    tts.save(mensaje_archivo)  # Guarda el mensaje como un archivo .mp3

    # Usar pygame para reproducir el archivo de audio
    pygame.mixer.music.load(mensaje_archivo)
    pygame.mixer.music.play()

    # Espera a que termine de reproducirse el mensaje
    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)  # Esperar a que termine la reproducción

    # Eliminar el archivo solo después de que termine la reproducción
    try:
        os.remove(mensaje_archivo)  # Elimina el archivo .mp3 después de reproducirlo
    except PermissionError:
        print("No se pudo eliminar el archivo, ya que está en uso.")

# Función que ejecuta el reconocimiento de audio en segundo plano
def escuchar_audio():
    global objetos_detectados  # Necesitamos que esta variable sea accesible globalmente

    while True:
        # Grabar y reconocer audio
        texto = reconocer_audio()

        # Verificar si "Mira." está en el texto
        if "mira." in texto.lower():  # Cambié "Mira." a "mira." para ser más flexible con la entrada
            mensaje = f"Se ha detectado un: {', '.join(objetos_detectados)}"
            print(mensaje)

            # Guardar el mensaje para que se procese por el TTS
            global mensaje_para_tts
            mensaje_para_tts = mensaje

            # Esperar un segundo para evitar la sobrecarga
            time.sleep(1)  # Pausa para evitar que el sistema esté demasiado sensible

# Función para verificar si hay un mensaje pendiente para TTS
def procesar_mensaje_tts():
    global mensaje_para_tts
    while True:
        if mensaje_para_tts:
            # Decir el mensaje en voz alta
            decir_mensaje(mensaje_para_tts)
            mensaje_para_tts = ""  # Limpiar el mensaje después de reproducirlo
        time.sleep(1)  # Esperar un segundo antes de verificar nuevamente

if __name__ == "__main__":
    # Crear tres hilos: uno para escuchar, otro para la detección de objetos y otro para el TTS
    hilo_grabacion = threading.Thread(target=grabar_audio)
    hilo_deteccion = threading.Thread(target=escuchar_y_responder)
    hilo_audio = threading.Thread(target=escuchar_audio)
    hilo_tts = threading.Thread(target=procesar_mensaje_tts)

    # Iniciar todos los hilos
    hilo_grabacion.start()
    hilo_deteccion.start()
    hilo_audio.start()
    hilo_tts.start()

    # Esperar que todos los hilos terminen
    hilo_grabacion.join()
    hilo_deteccion.join()
    hilo_audio.join()
    hilo_tts.join()
