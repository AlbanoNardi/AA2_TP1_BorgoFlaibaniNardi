"""
Script 3: Prueba del sistema completo (rock-paper-scissors.py)
Este script carga un modelo previamente entrenado para clasificar gestos de "piedra, papel o tijeras",
captura imágenes de la cámara web en tiempo real, detecta landmarks de manos mediante MediaPipe
y muestra la predicción del gesto reconocido en la pantalla.
"""

# Importación de bibliotecas necesarias
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time

# Definir el umbral de confianza (70%)
CONFIDENCE_THRESHOLD = 0.7

# Cargar el modelo entrenado
try:
    model = keras.models.load_model('model5411param.h5')
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Diccionario de clases
class_names = ["Piedra", "Papel", "Tijeras"]

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
            
    # Espejar la imagen para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Convertir BGR a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    cv2.putText(frame,"Presiona q para salir",(400, 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),2)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar los landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer las coordenadas x, y normalizadas de los 21 puntos
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Convertir a NumPy array
            landmarks_array = np.array(landmarks).reshape(1, -1)

            # Predecir con el modelo
            prediction = model.predict(landmarks_array)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            probabilidad = confidence * 100
            
            # Mostrar información en consola
            print(f"Clase predicha: {class_names[class_id]} - Probabilidad: {probabilidad.round(2)}%")

            # Verificar si la confianza supera el umbral
            if confidence >= CONFIDENCE_THRESHOLD:
                class_name = class_names[class_id]
                # Mostrar la predicción en pantalla
                cv2.putText(frame, f"Gesto: {class_name} ({probabilidad:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Mostrar mensaje de baja confianza
                cv2.putText(frame, "Confianza baja", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Piedra, Papel o Tijeras", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()