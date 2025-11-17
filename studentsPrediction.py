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
    page_title="Logistic Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título de la página
st.title("Logistic Regression Model for Students Performance")

# Cargar el Dataset
data = pd.read_csv('StudentsPerformance.csv')
data = data.drop('lunch', axis=1)  # 🔹 Eliminar lunch

# Codificar numéricamente las columnas object
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
option = st.sidebar.selectbox("Select an option", ("Data preview", "Data summary", "Confusion matrix", "Classification Report", "ROC Accuracy", "Predict"))

if option == "Data preview":
    st.write("### Data Preview", data.head())
    
elif option == "Data summary":
    st.write("### Data summary", data.describe())

elif option == "Confusion matrix":
    st.header("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

elif option == "Classification Report":
    st.header("Classification Report")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report)

elif option == "ROC Accuracy":
    st.header("ROC Accuracy")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f'AUC: {auc}')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

elif option == "Predict":
    st.header("Predict for a student")

    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox("Parental level of education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    math_score = st.slider("Math score", 0, 100, 50)
    reading_score = st.slider("Reading score", 0, 100, 50)
    writing_score = st.slider("Writing score", 0, 100, 50)

    if st.button("Predict"):
        prediction = predict_student(gender, race, parental_education, math_score, reading_score, writing_score)
        if prediction == 1:
            st.success("The student has completed the test preparation course")
        else:
            st.error("The student has not completed the test preparation course")
