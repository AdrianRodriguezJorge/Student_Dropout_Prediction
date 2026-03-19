# 🎓 Predicción de Deserción Estudiantil mediante Regresión Logística

## 📘 Descripción General
Este proyecto desarrolla un modelo de *machine learning* orientado a predecir la probabilidad de que un estudiante abandone sus estudios antes de completarlos. A partir de información académica y socioeconómica, se entrena un modelo de **Regresión Logística** capaz de identificar patrones asociados al riesgo de deserción.

La finalidad es proporcionar una herramienta que permita a instituciones educativas **detectar casos de riesgo de forma temprana** y aplicar estrategias de acompañamiento más efectivas.

---

## 🎯 Objetivo del Proyecto
- Construir un modelo predictivo basado en características del estudiante.  
- Analizar factores que influyen en la deserción académica.  
- Facilitar la toma de decisiones mediante una interfaz interactiva de predicción.  

---

## 🧠 Metodología
El desarrollo del proyecto sigue las siguientes etapas:

### 🔍 1. Análisis y preparación de datos
- Limpieza del dataset.  
- Codificación y normalización de variables.  
- Manejo del desbalance de clases mediante **SMOTE**.

### 🤖 2. Entrenamiento del modelo
- Implementación de un modelo de **Regresión Logística**.  
- Ajuste de hiperparámetros y validación cruzada.

### 📈 3. Evaluación
Se emplean métricas clave para medir el rendimiento:
- *Accuracy*  
- *Precision*  
- *Recall*  
- *ROC-AUC*

### 🖥️ 4. Interfaz de predicción
Se implementa una aplicación en **Streamlit** que permite ingresar características del estudiante y obtener una predicción inmediata.

---

## 🗂️ Estructura del Proyecto
| Archivo | Descripción |
|--------|-------------|
| **StudentsPerformance.csv** | Dataset utilizado para el entrenamiento y evaluación. |
| **model_with_smote.ipynb** | Notebook con el proceso completo de modelado y balanceo de clases. |
| **studentsPrediction.py** | Aplicación en Streamlit para realizar predicciones. |

---

## ▶️ Uso de la Aplicación
1. Ejecuta el script de Streamlit.  
2. Accede a la URL local que se mostrará en consola.  
3. Introduce los datos del estudiante en el formulario.  
4. Obtén la predicción del modelo en tiempo real.

---

## 📌 Mejoras Futuras
- Evaluar modelos adicionales (Random Forest, XGBoost, Redes Neuronales).  
- Incorporar técnicas de explicabilidad como **SHAP** o **LIME**.  
- Desplegar el modelo mediante una API.  
- Ampliar el dataset con nuevas variables relevantes.

---

## ⚙️ Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear un entorno virtual (opcional)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
streamlit run studentsPrediction.py
```

---

https://img.shields.io/badge/Python-3.10-blue
https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-green
https://img.shields.io/badge/Estado-Activo-success
https://img.shields.io/badge/Licencia-MIT-yellow
