"""
Script para el entrenamiento de un clasificador de gestos "piedra", "papel" o "tijeras"
utilizando una red neuronal densa para procesar los landmarks detectados por MediaPipe.
"""

## Preparación del entorno

# Importación de librerías necesarias
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import io

## Análisis Exploratorio

# Urls de los dataset creados con MediaPipe

url_rps_dataset_f = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_flaibani.npy'
url_rps_labels_f = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_flaibani.npy'

url_rps_dataset_b = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_borgo.npy'
url_rps_labels_b = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_borgo.npy'

url_rps_dataset_n = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_nardi.npy'
url_rps_labels_n = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_nardi.npy'

url_rps_dataset_ni = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_nardii.npy'
url_rps_labels_ni = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_nardii.npy'

url_rps_dataset_fi = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_flaibanii.npy'
url_rps_labels_fi = 'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_flaibanii.npy'

# Función para descargar y cargar archivos .npy desde URLs

def load_npy_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return np.load(io.BytesIO(response.content))
    else:
        raise Exception(f"Error al descargar el archivo: {response.status_code}")

x_flaibani = load_npy_from_url(url_rps_dataset_f)
y_flaibani = load_npy_from_url(url_rps_labels_f)

x_borgo = load_npy_from_url(url_rps_dataset_b)
y_borgo = load_npy_from_url(url_rps_labels_b)

x_nardi = load_npy_from_url(url_rps_dataset_n)
y_nardi = load_npy_from_url(url_rps_labels_n)

x_nardii = load_npy_from_url(url_rps_dataset_n)
y_nardii = load_npy_from_url(url_rps_labels_n)

x_flaibanii = load_npy_from_url(url_rps_dataset_f)
y_flaibanii = load_npy_from_url(url_rps_labels_f)

# Verificación de dimensiones de los datasets

print(x_flaibani.shape, y_flaibani.shape)
print(x_borgo.shape, y_borgo.shape)
print(x_nardi.shape, y_nardi.shape)
print(x_flaibanii.shape, y_flaibanii.shape)
print(x_nardii.shape, y_nardii.shape)

# Concatenación y guardado de los datasets

X_data = np.concatenate((x_flaibani, x_borgo, x_nardi, x_flaibanii, x_nardii), axis=0)
y_labels = np.concatenate((y_flaibani, y_borgo, y_nardi, y_flaibanii, y_nardii), axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels,
)

# Early stop para el modelo
early_stop = EarlyStopping(
        monitor='val_loss',  # Monitorea la pérdida en validación
        patience=15,         # Número de épocas sin mejora antes de detener
        verbose=0,
        restore_best_weights=True,  # Restaura los mejores pesos encontrados
        mode='min',          # Minimizar la pérdida
    )

model = Sequential([
    Input(shape=(X_train.shape[1],)),   # Capa de entrada: tamaño basado en el número de características (42 valores: x,y para 21 landmarks)
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(3, activation="softmax")
])

model.summary() 

# Compilación del modelo
model.compile(
        optimizer='adam',  # Optimizador adaptativo
        loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación con etiquetas enteras
        metrics=['accuracy']  # Métrica para monitorear
)

history = model.fit(
        X_train, y_train,
        epochs=1500,             # Número máximo de épocas
        batch_size=16,           # Tamaño del lote
        validation_split=0.2,    # 20% de los datos de entrenamiento para validación
        callbacks=[early_stop],  # Callback para detener el entrenamiento si no hay mejora
)

model.save('model5411param.h5')

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Precisión en conjunto de prueba: {test_accuracy:.4f}")
print(f"Pérdida en conjunto de prueba: {test_loss:.4f}")

# Después de entrenar
plt.figure(figsize=(12, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Curvas de Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Curvas de Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()