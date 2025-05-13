import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import io
import time

# URLs de los datasets
urls = {
    "x": [
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_flaibani.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_borgo.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_nardi.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_nardii.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_dataset_flaibanii.npy'
    ],
    "y": [
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_flaibani.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_borgo.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_nardi.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_nardii.npy',
        'https://raw.githubusercontent.com/AlbanoNardi/AA2_TP1_BorgoFlaibaniNardi/main/ejercicio_2/rps_labels_flaibanii.npy'
    ]
}

# Función para cargar .npy desde URL
def load_npy_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        time.sleep(1)
        return np.load(io.BytesIO(response.content))
    else:
        raise Exception(f"Error al descargar {url}")

# Cargar todos los datasets
X_data = np.concatenate([load_npy_from_url(u) for u in urls["x"]], axis=0)
y_labels = np.concatenate([load_npy_from_url(u) for u in urls["y"]], axis=0)

print("Dataset original:", X_data.shape, y_labels.shape)

# Data Augmentation para landmarks (coordenadas x, y normalizadas)
def augment_landmarks(X, y, num_aug=2):
    augmented_X, augmented_y = [], []

    for i in range(len(X)):
        sample = X[i]
        label = y[i]
        for _ in range(num_aug):
            augmented = sample.copy().reshape(-1, 2)

            # Agregar ruido gaussiano
            augmented += np.random.normal(0, 0.01, size=augmented.shape)

            # Rotación leve
            angle = np.random.uniform(-10, 10) * np.pi / 180
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle),  np.cos(angle)]])
            augmented = np.dot(augmented, rot_matrix)

            # Escalado leve
            scale = np.random.uniform(0.9, 1.1)
            augmented *= scale

            # Traslación leve
            shift = np.random.uniform(-0.02, 0.02, size=(1, 2))
            augmented += shift

            augmented_X.append(augmented.flatten())
            augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)

# Aplicar data augmentation
X_aug, y_aug = augment_landmarks(X_data, y_labels, num_aug=2)
X_full = np.concatenate([X_data, X_aug], axis=0)
y_full = np.concatenate([y_labels, y_aug], axis=0)

print("Dataset con augmentation:", X_full.shape, y_full.shape)

# Dividir conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Callback
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min')

# Modelo denso
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    X_train, y_train,
    epochs=1500,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Guardar modelo
model.save('model_augmented.h5')

# Evaluación
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en conjunto de prueba: {test_accuracy:.4f}")
print(f"Pérdida en conjunto de prueba: {test_loss:.4f}")

# Gráficas
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Curvas de Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Curvas de Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.tight_layout()
plt.show()
