# Clasificador de Gestos - Piedra, Papel o Tijeras ✊✋✌️

## Trabajo Práctico - Redes Densas y Convolucionales 2025
**Tecnicatura Universitaria en Inteligencia Artificial**

### Profesores
* Moreyra, Matias
* Cocitto López, Bruno
* Moreyra, Facundo

### Integrantes
* Borgo Elgart, Iair (Legajo: B-6608/7)
* Flaibani, Marcela (Legajo: F-3793/1)
* Nardi, Albano (Legajo: N-1280/7)

## 📌 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de gestos de mano para el juego "Piedra, Papel o Tijeras" utilizando técnicas de aprendizaje automático. Se emplea **MediaPipe** para la detección de puntos clave (landmarks) de la mano, y una red neuronal completamente conectada para clasificar el gesto mostrado por el usuario.

El desarrollo se organiza en tres scripts:

- `record-dataset.py`: Captura y guarda muestras de gestos usando la webcam.
- `train-gesture-classifier.py`: Entrena un modelo de red neuronal sobre los datos capturados.
- `rock-paper-scissors.py`: Clasifica gestos en tiempo real desde la cámara.

---

## 📁 Estructura del Proyecto

ejercicio_2/
├── record-dataset.py                # Captura de gestos mediante webcam
├── train-gesture-classifier.py     # Entrenamiento de la red neuronal
├── rock-paper-scissors.py          # Clasificación en tiempo real
├── model5411param.h5               # Modelo entrenado
├── graf_model_5411.png             # Curvas de entrenamiento (loss/accuracy)
├── rps_dataset_*.npy               # Archivos de datos (features)
├── rps_labels_*.npy                # Archivos de etiquetas (labels)
└── imagenes/
    ├── capturas_dataset/           # Imágenes capturadas durante la grabación del dataset
    └── pruebas_prediccion/        # Capturas de pruebas en tiempo real



---

## 🧰 Tecnologías Utilizadas

- Python 3.10+
- Python 3.12+
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://developers.google.com/mediapipe)
- [NumPy](https://numpy.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

---

## 📦 Dataset

El dataset fue generado utilizando MediaPipe con python en la versión "Python 3.10" ya que las posteriores presentaron problemas, extrayendo 21 puntos clave por mano, resultando en 42 coordenadas (x, y) por muestra.

- Se capturaron gestos de múltiples personas.
- Se variaron posiciones, ángulos y condiciones de luz.
- Las clases utilizadas fueron:
  - `0`: Piedra ✊
  - `1`: Papel ✋
  - `2`: Tijeras ✌️

Las muestras están almacenadas en los archivos:

### Features:
- rps_dataset_borgo.npy
- rps_dataset_flaibani.npy
- rps_dataset_nardi.npy
- rps_dataset_flaibanii.npy
- rps_dataset_nardii.npy

### Etiquetas:
- rps_labels_borgo.npy
- rps_labels_flaibani.npy
- rps_labels_nardi.npy
- rps_labels_flaibanii.npy
- rps_labels_nardii.npy

---

## 🧠 Entrenamiento del Modelo

El modelo entrenado es una red neuronal densa con tres capas ocultas. Se utilizaron técnicas de regularización (dropout) y early stopping para evitar el sobreajuste.

- Arquitectura: fully connected con ReLU.
- Optimización: Adam.
- Métrica: Accuracy.
- Se guardó el modelo final como `model5411param.h5`.

---

## 🧪 Evaluación

Durante las pruebas en tiempo real (`rock-paper-scissors.py`):

- Se utilizó la webcam para detectar gestos.
- Se aplicó un **umbral de confianza (0.7)** para validar predicciones.
- La predicción aparece sobre el video si la confianza es suficiente.

Capturas de predicciones y dataset están disponibles en la carpeta `imagenes/`.
Captura de gráfica con curvas de aprendizaje disponible en 'graf_model.png'

---

## ✅ Resultados

- El modelo logró una precisión validada mayor al **85%**.
- El rendimiento fue robusto ante variaciones de posición.
- La confianza de predicción superó el 85% en la mayoría de las muestras bien enfocadas.

---

## 🎓 Conclusión

> Pudimos integrar una herramienta de visión por computadora como MediaPipe con redes neuronales densas, logrando buenos resultados en tiempo real.
>
> El modelo demostró una gran capacidad de generalización, lo cual atribuyo a la diversidad de datos recolectados: distintas manos, ángulos y posiciones. Esto fue intencional desde el diseño, pensando en la escalabilidad del sistema y su posible adaptación a un entorno real.
>
> Además, trabajar con un umbral de confianza fue útil para asegurar que el sistema no hiciera predicciones erróneas cuando los gestos no eran claros.
>

---

## 📝 Créditos

Trabajo realizado para la materia **Aprendizaje Automático 2**, correspondiente a la **Tecnicatura Universitaria en Inteligencia Artificial**.

---
