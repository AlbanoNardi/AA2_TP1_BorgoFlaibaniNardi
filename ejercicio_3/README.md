# 🖼️ Clasificación de Imágenes Naturales con Redes Neuronales 🧠🌍

## Trabajo Práctico - Clasificación de Imágenes con Redes Neuronales 2025

**Tecnicatura Universitaria en Inteligencia Artificial**

### Profesores

* Moreyra, Matias
* Cocitto López, Bruno
* Moreyra, Facundo

### Integrantes

* Borgo Elgart, Iair (Legajo: B-6608/7)
* Flaibani, Marcela (Legajo: F-3793/1)
* Nardi, Albano (Legajo: N-1280/7)

---

## 📌 Descripción del Problema

Este trabajo corresponde al **Ejercicio 3** del primer trabajo práctico de la materia **Aprendizaje Automático II**. El objetivo principal es desarrollar y comparar diferentes modelos de clasificación de imágenes utilizando redes neuronales densas, convolucionales, residuales y con aprendizaje por transferencia (transfer learning).

Se utiliza un conjunto de imágenes de escenas naturales capturadas en distintas partes del mundo, que deben clasificarse en una de las seis categorías siguientes:

* `buildings`
* `forest`
* `glacier`
* `mountain`
* `sea`
* `street`

---

## 🧾 Dataset

El conjunto de datos utilizado se encuentra disponible en el siguiente enlace:
📁 [TP1-3-natural-scenes.zip](https://drive.google.com/file/d/1Pqs5Y6dZr4R66Dby5hIUIjPZtBI28rmJ/view?usp=drive_link)

Estructura del dataset:

* `seg_train/`: \~14,000 imágenes para entrenamiento
* `seg_test/`: \~3,000 imágenes para evaluación del modelo
* `seg_pred/`: \~7,000 imágenes para predicción final

Todas las imágenes tienen un tamaño de **150x150 píxeles** y están etiquetadas según una de las seis clases mencionadas.

---

## ⚙️ Ejecución en Google Colab

1. Abrir el notebook `TP1-AAII-1C-2025_Ej3_BorgoFlaibaniNardi.ipynb`.
2. Ejecutar todas las celdas del notebook para:

   * Descargar y descomprimir el dataset
   * Explorar visualmente el conjunto de datos
   * Preprocesar imágenes y aplicar aumentación
   * Construir, entrenar y evaluar los distintos modelos
   * Visualizar métricas de desempeño y predicciones

---

## 🧠 Modelos Desarrollados

### 🔹 Modelo Denso (Dense)

* Arquitectura completamente conectada sin convoluciones
* Precisión en validación: **54%**
* Número de parámetros: \~8.6M
* Buen punto de partida, pero limitado para imágenes

### 🔹 Modelo Convolucional

* Arquitectura: `Conv2D + MaxPooling + Dense`
* Precisión en validación: **\~86%**
* Parámetros: \~148K
* Balance ideal entre precisión y tamaño

### 🔹 Modelo con Bloques Residuales (Identidad)

* Uso de `Add()` y conexiones tipo ResNet
* Precisión en validación: **\~75%**
* Parámetros: \~552K
* Riesgo de sobreajuste sin aumento de datos

### 🔹 Modelo con Backbone (Transfer Learning)

* Utiliza `MobileNetV3Small` con pesos de ImageNet
* Capa de clasificación personalizada
* Muy buena precisión, aunque muestra **overfitting**
* Rápido entrenamiento gracias a congelación de pesos

---

## 📊 Evaluación y Visualización

* Gráficas de precisión y pérdida para entrenamiento y validación.
* Comparación visual entre predicciones y clases reales.
* Evaluación con imágenes no vistas (conjunto `seg_pred`).
* Análisis del rendimiento de cada arquitectura.

---

## 🎓 Conclusión


---

## 📝 Créditos

Trabajo realizado para la materia **Aprendizaje Automático 2**, correspondiente a la **Tecnicatura Universitaria en Inteligencia Artificial**, 1º cuatrimestre 2025.

---
