import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE  

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Regresión Logística",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título de la página
st.title("Modelo de Regresión Logística para el Rendimiento Estudiantil")

# Cargar el Dataset
data = pd.read_csv('StudentsPerformance.csv')
data = data.drop('lunch', axis=1)  # 🔹 Eliminar lunch

# Codificar numéricamente las columnas categóricas
encoder = OrdinalEncoder()
columns_to_encode = ['gender', 'race/ethnicity', 'parental level of education']
encoder.fit(data[columns_to_encode])
data[columns_to_encode] = encoder.transform(data[columns_to_encode])

# Dividir datos en entrenamiento y prueba  
X = data[['gender', 'race/ethnicity', 'parental level of education', 'math score', 'reading score', 'writing score']]
y = data['test preparation course']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balancear los datos con SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

# Predicción
y_pred = model.predict(X_test)

# Diccionario de mapeo para mostrar etiquetas más claras
race_mapping = {
    "group A": "Afrodescendiente",
    "group B": "Caucásico",
    "group C": "Hispano",
    "group D": "Asiático",
    "group E": "Otro"
}

# Función para predecir si el estudiante ha completado el curso
def predict_student(gender, race, parental_education, math_score, reading_score, writing_score):
    # Crear DataFrame con el nuevo estudiante
    student = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [parental_education],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })

    # Codificar usando el mismo encoder entrenado
    student[columns_to_encode] = encoder.transform(student[columns_to_encode])

    # Asegurar que las columnas coincidan con las usadas en el entrenamiento
    student = student[X.columns]

    # Realizar la predicción
    prediction = model.predict(student)
    return prediction[0]

# Opciones de la interfaz
option = st.sidebar.selectbox(
    "Selecciona una opción", 
    ("Vista previa de datos", "Resumen de datos", "Matriz de confusión", "Reporte de clasificación", "Curva ROC", "Predecir")
)

if option == "Vista previa de datos":
    st.write("### Vista previa de los datos", data.head())
    
elif option == "Resumen de datos":
    st.write("### Resumen estadístico de los datos", data.describe())

elif option == "Matriz de confusión":
    st.header("Matriz de confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)

elif option == "Reporte de clasificación":
    st.header("Reporte de clasificación")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report)

elif option == "Curva ROC":
    st.header("Curva ROC y AUC")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f'AUC: {auc}')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)

elif option == "Predecir":
    st.header("Predicción para un estudiante")

    gender = st.selectbox("Sexo", ["male", "female"], format_func=lambda x: "Masculino" if x=="male" else "Femenino")
    race = st.selectbox("Raza/Etnia", list(race_mapping.keys()), format_func=lambda x: race_mapping[x])
    parental_education = st.selectbox("Nivel educativo de los padres", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ], format_func=lambda x: {
        "some high school": "Algo de secundaria",
        "high school": "Secundaria completa",
        "some college": "Algo de universidad",
        "associate's degree": "Técnico/Asociado",
        "bachelor's degree": "Licenciatura",
        "master's degree": "Maestría"
    }[x])
    math_score = st.slider("Nota en Matemáticas", 0, 100, 50)
    reading_score = st.slider("Nota en Lectura", 0, 100, 50)
    writing_score = st.slider("Nota en Escritura", 0, 100, 50)

    if st.button("Predecir"):
        prediction = predict_student(gender, race, parental_education, math_score, reading_score, writing_score)
        if prediction == 1:
            st.success("El estudiante ha completado el curso de preparación para exámenes")
        else:
            st.error("El estudiante NO ha completado el curso de preparación para exámenes")
