# 📘 Predicción del Rendimiento Académico con Redes Neuronales 🎓📈

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

---

## 📌 Descripción del Proyecto

Este proyecto corresponde al **Ejercicio 1** del TP y tiene como objetivo desarrollar un modelo de regresión basado en redes neuronales para predecir el **índice de rendimiento académico** de estudiantes universitarios.  

Se utilizó un conjunto de datos que contiene información sobre hábitos de estudio, sueño, prácticas y actividades extracurriculares, junto con la calificación obtenida como variable objetivo.

---

## 🧾 Dataset

El dataset original se encuentra disponible en [Google Drive](https://drive.google.com/file/d/1mfpXVLqDJah-sO0CF29LjKUz5NtKjZqc/view?usp=drive_link) y fue cargado directamente desde GitHub en formato `.csv`.

Variables disponibles:
- `hr_studied`: Horas dedicadas al estudio por día.
- `prev_score`: Puntajes obtenidos previamente.
- `extra_act`: Participación en actividades extracurriculares (sí/no).
- `hr_sleep`: Horas de sueño promedio por día.
- `practice_count`: Cuestionarios de práctica realizados.
- `y`: Índice de rendimiento académico (target, entre 10 y 100).

---

## ⚙️ Ejecución en Google Colab

1. Descargar y abrir desde el entorno colab el fichero `TP1-AAII-1C-2025_Ej1_BorgoFlaibaniNardi.ipynb`.
2. Ejecutar las celdas en orden para:
   - Cargar y explorar el dataset
   - Preprocesar los datos (normalización, codificación)
   - Definir y entrenar el modelo
   - Evaluar su desempeño

---

## 🧠 Modelo y Entrenamiento

- Se utilizaron únicamente las variables `prev_score` y `hr_studied`, tras el EDA.
- Modelo: red neuronal densa de 2 capas ocultas:
  - `Dense(32, activation='sigmoid')`
  - `Dense(16, activation='sigmoid')`
- Capa de salida: `Dense(1)`
- Pérdida: `mse` (error cuadrático medio)
- Métrica: `mae` (error absoluto medio)
- Optimizador: `adam`
- Épocas: 20
- Total de parámetros: 641

---

## 📊 Resultados

- **MAE (escala original)**: 1.87 puntos
- **R² (coeficiente de determinación)**: 0.98  
- **MSE (validación)**: ~0.00067

Las predicciones fueron visualizadas y comparadas con los valores reales mediante un scatter plot, mostrando un alineamiento muy próximo a la diagonal ideal.

---

## 🧪 Evaluación y Visualización

- Se realizaron gráficos de pérdida y error por época.
- Se graficó la correlación entre variables.
- Las variables seleccionadas mostraron una correlación positiva con el rendimiento académico (`prev_score` y `hr_studied`).

---

## 🎓 Conclusión

> En este trabajo se desarrolló un modelo de regresión utilizando una red neuronal para predecir el rendimiento académico de estudiantes universitarios, basándose únicamente en las variables `prev_score` y `hr_studied`, que fueron las más relevantes según el análisis exploratorio.
>
> El modelo entrenado fue simple pero efectivo, logrando una precisión muy alta. La métrica R²=0.98 sugiere que el modelo explica prácticamente toda la variabilidad del rendimiento académico con solo dos variables.
>
> El error absoluto medio en escala original (MAE ≈ 1.87) indica una alta precisión en la predicción. Los gráficos de entrenamiento y validación mostraron curvas estables, sin evidencia de sobreajuste.
>
> En resumen, el modelo cumple satisfactoriamente con el objetivo planteado y representa una aplicación clara y concreta del uso de redes neuronales densas en problemas de regresión.

---

## 📝 Créditos

Trabajo realizado para la materia **Aprendizaje Automático 2**, correspondiente a la **Tecnicatura Universitaria en Inteligencia Artificial**, 1° cuatrimestre 2025.
