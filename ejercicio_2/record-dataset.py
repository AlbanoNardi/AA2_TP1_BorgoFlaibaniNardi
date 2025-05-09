"""
Script para la grabación de un dataset de gestos de "piedra", "papel" o "tijeras"
utilizando MediaPipe para la detección de landmarks de la mano.

Este script captura datos de entrenamiento para un clasificador de gestos RPS (Rock-Paper-Scissors)
usando la cámara web y MediaPipe para la detección de puntos clave de la mano.

# Requisitos: pip install opencv-python mediapipe numpy
"""

# Importación de librerías necesarias
import cv2
import mediapipe as mp
import numpy as np
import os
import time  # Para añadir retrasos y controlar la frecuencia de captura

def main():

    # Modelo de detección de manos Se incrementael umbral de detección (por defecto 0.5) y se determina máximo 1 mano
    mp_hands = mp.solutions.hands                                                                   
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,min_detection_confidence=0.7)   
    mp_drawing = mp.solutions.drawing_utils

    # Crear listas para almacenar los datos y etiquetas
    X_data = []
    y_labels = []

    # Diccionario de mapeo de etiquetas: 0 = piedra, 1 = papel, 2 = tijeras
    labels_dict = {0: "piedra", 1: "papel", 2: "tijeras"}
    
    # Contador de muestras por clase para mostrar al usuario
    samples_count = {0: 0, 1: 0, 2: 0}

    # Inicializar la captura de video desde la webcam (0 es generalmente la webcam integrada)
    cap = cv2.VideoCapture(0)
    
    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara. Verifica que esté conectada y disponible.")
        return

    print("Instrucciones:")
    print("- Muestra tu mano frente a la cámara")
    print("- Presiona las siguientes teclas para grabar gestos:")
    print("  * 0: para grabar gesto de 'piedra'")
    print("  * 1: para grabar gesto de 'papel'")
    print("  * 2: para grabar gesto de 'tijeras'")
    print("  * q: para salir y guardar el dataset")
    print("  * d: para eliminar la última muestra guardada")
    print("\nComenzando captura de video. Presiona 'q' para salir.")
    
    last_captured = None  # Variable para almacenar la información del último gesto capturado

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)                      # Espejar la imagen para vista natural (izquierda-derecha)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Convertir de BGR (OpenCV) a RGB (requerido por MediaPipe)
        result = hands.process(rgb)                     # Procesar el frame con MediaPipe para detectar manos

        # Copia del frame para mostrar información
        display_frame = frame.copy()
        
        # Si se detectan manos, extraer los landmarks
        landmark_list = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Dibujar los landmarks en el frame
                mp_drawing.draw_landmarks(
                    display_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS)

                # Extraer coordenadas normalizadas (x, y) de los landmarks
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y])  # Concatenar coordenadas x, y

        # Mostrar instrucciones y contadores en pantalla
        cv2.putText(display_frame,"Presiona: 0=piedra, 1=papel, 2=tijeras, q=salir, d=eliminar ultimo",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),2)
        
        # Mostrar contadores de muestras
        y_pos = 60
        for label, name in labels_dict.items():
            cv2.putText(
                display_frame,
                f"{name}: {samples_count[label]} muestras", 
                (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 0), 
                2
            )
            y_pos += 30
            
        # Si hay un último gesto capturado, mostrarlo
        if last_captured:
            cv2.putText(
                display_frame,
                f"Ultimo capturado: {last_captured}", 
                (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )

        # Mostrar el frame con la información
        cv2.imshow("Grabando dataset de gestos RPS", display_frame)

        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Salir
            break

        elif key in [ord('0'), ord('1'), ord('2')]:             # Si 0, 1 o 2, guardar muestra con su etiqueta
            if result.multi_hand_landmarks and landmark_list:   # Verificar que haya una mano detectada
                label = int(chr(key))
                X_data.append(landmark_list)
                y_labels.append(label)
                samples_count[label] += 1
                last_captured = f"{labels_dict[label]} (total: {samples_count[label]})"
                print(f"Gesto guardado: {labels_dict[label]} - Total de esta clase: {samples_count[label]}")
                
                time.sleep(0.5)                                 # Pequeña pausa para evitar capturas múltiples accidentales
            else:
                print("No se detecto ninguna mano. Acerca tu mano a la camara.")
                
        elif key == ord('d'):  # Eliminar última muestra
            if X_data and y_labels:
                last_label = y_labels.pop()
                X_data.pop()
                samples_count[last_label] -= 1
                print(f"Ultima muestra eliminada: {labels_dict[last_label]}")
                
                # Actualizar información del último gesto capturado
                if X_data and y_labels:
                    last_label = y_labels[-1]
                    last_captured = f"{labels_dict[last_label]} (total: {samples_count[last_label]})"
                else:
                    last_captured = None
            else:
                print("No hay muestras para eliminar.")

    # Guardar el dataset solo si hay datos capturados
    if X_data and y_labels:
        # Convertir listas a arrays de NumPy
        X_array = np.array(X_data)
        y_array = np.array(y_labels)
        
        # Guardar arrays en archivos .npy
        np.save("rps_dataset.npy", X_array)
        np.save("rps_labels.npy", y_array)
        
        print("\nDataset guardado exitosamente:")
        print(f"- Archivo de datos: rps_dataset.npy - Shape: {X_array.shape}")
        print(f"- Archivo de etiquetas: rps_labels.npy - Shape: {y_array.shape}")
        print("\nResumen de muestras por clase:")
        total_samples = sum(samples_count.values())
        for label, name in labels_dict.items():
            percentage = (samples_count[label] / total_samples) * 100 if total_samples > 0 else 0
            print(f"- {name}: {samples_count[label]} muestras ({percentage:.1f}%)")
    else:
        print("No se guardo ningun dato porque no se capturaron muestras.")

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()