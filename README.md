# 🎓 Student Dropout Prediction using Logistic Regression

## 📘 Overview
Este proyecto aplica técnicas de *machine learning* para predecir la probabilidad de que un estudiante abandone sus estudios antes de completarlos. Utilizando un modelo de **Regresión Logística**, se analizan patrones en los datos académicos y socioeconómicos para identificar factores asociados al riesgo de deserción.

## 🎯 Objective
El objetivo principal es construir un modelo capaz de estimar, a partir de características del estudiante (rendimiento académico, nivel educativo de los padres, entre otros), si existe riesgo de abandono.  
Esta información puede ayudar a instituciones educativas a **intervenir tempranamente** y ofrecer apoyo personalizado.

## 🗂️ Project Structure
| Archivo | Descripción |
|--------|-------------|
| **StudentsPerformance.csv** | Dataset utilizado para entrenar el modelo. |
| **model_with_smote.ipynb** | Notebook con el proceso de entrenamiento, incluyendo balanceo de clases mediante **SMOTE**. |
| **studentsPrediction.py** | Interfaz desarrollada en **Streamlit** para realizar predicciones de forma interactiva. |

## 🧠 Methodology
- Limpieza y preprocesamiento del dataset.  
- Balanceo de clases con **SMOTE** para mejorar el rendimiento del modelo.  
- Entrenamiento de un modelo de **Regresión Logística**.  
- Evaluación mediante métricas como *accuracy*, *precision*, *recall* y *ROC-AUC*.  
- Implementación de una interfaz sencilla para predicciones en tiempo real.

## 🚀 Streamlit App
La aplicación permite ingresar características del estudiante y obtener una predicción inmediata sobre su riesgo de abandono.

`https://github.com/user-attachments/assets/b3946d82-1951-43ec-89ed-9cf62521eaca`

## 📌 Future Improvements
- Probar modelos adicionales (Random Forest, XGBoost, Redes Neuronales).  
- Añadir explicabilidad del modelo (SHAP, LIME).  
- Integrar una API para despliegue en producción.  
- Ampliar el dataset con nuevas variables relevantes.

---

## ⚙️ Instalación

Sigue estos pasos para ejecutar el proyecto en tu entorno local:

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear un entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación Streamlit
```bash
streamlit run studentsPrediction.py
```

---

## ▶️ Uso

1. Abre la aplicación en tu navegador (Streamlit mostrará la URL, normalmente `http://localhost:8501`).
2. Introduce las características del estudiante en el formulario.
3. El modelo generará una predicción indicando si existe riesgo de abandono.
4. Ajusta los parámetros y experimenta con diferentes perfiles.

---

## 🏷️ Badges para tu README

Aquí tienes algunos badges listos para usar. Puedes mezclarlos o elegir los que mejor encajen con tu estilo:

### ⭐ Badges básicos

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-green)

### 🧪 Badges de estado del proyecto

![Status](https://img.shields.io/badge/Status-Active-success)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen)

### 📊 Badges de licencia y contribuciones

![License](https://img.shields.io/badge/License-MIT-yellow)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

### 🧹 Badges estéticos
![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4)
![Clean Code](https://img.shields.io/badge/Clean-Code-blueviolet)
