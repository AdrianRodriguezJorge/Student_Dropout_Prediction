import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.over_sampling import SMOTE  

st.set_page_config(
    page_title="Logistic Regression",
    layout="wide", #centered
    initial_sidebar_state="expanded",
)
# Título de la página
st.title("Logistic Regression Model for stundents performance")

# Función para predecir si el estudiante seguirá en la carrera o no
def predict_student(gender, race, parental_education, lunch, math_score, reading_score, writing_score):
    aux = pd.read_csv('StudentsPerformance.csv')
    aux = pd.concat([aux, pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [parental_education],
        'lunch': [lunch],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score],
        'studying': [0]
    })])

    # codificar numericamente las columnas Object
    encoder = OrdinalEncoder()

    columns_to_encode = ['gender', 'race/ethnicity', 'parental level of education', 'lunch']

    encoder.fit(aux[columns_to_encode])

    aux[columns_to_encode] = encoder.transform(aux[columns_to_encode])

    student = aux.iloc[1000]
    student = student.drop('studying')  
    
    X_student = pd.DataFrame([student])  
    
    prediction = model.predict(X_student)
    
    return prediction[0]


############################## Cargando el Dataset
data = pd.read_csv('StudentsPerformance.csv')

# codificar numericamente las columnas object
encoder = OrdinalEncoder()

columns_to_encode = ['gender', 'race/ethnicity', 'parental level of education', 'lunch']

encoder.fit(data[columns_to_encode])

data[columns_to_encode] = encoder.transform(data[columns_to_encode])

# Dividir datos en entrenamiento y prueba  
X = data[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'math score', 'reading score', 'writing score']]
y = data[['studying']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balancear los datos con Smote
smote = SMOTE(random_state=42)  
X_resampled, y_resampled = smote.fit_resample(X_train, y_train) # type: ignore

# Aplicar RandomUnderSampler  
undersampler = RandomUnderSampler(random_state=42)  
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)   # type: ignore

# Inicializar y entrenar el modelo de regresión logística  
model = LogisticRegression()  
model.fit(X_resampled, y_resampled)

# Predicción
y_pred = model.predict(X_test)


############################## Opciones de la interfaz
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
    st.write(auc)

elif option == "Predict":
    st.header("Predict for a student")

    # Ingreso de datos del estudiante
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox("Parental level of education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Type of lunch", ["standard", "free/reduced"])
    math_score = st.slider("Math score", 0, 100, 50)
    reading_score = st.slider("Reading score", 0, 100, 50)
    writing_score = st.slider("Writing score", 0, 100, 50)

    if st.button("Predict"):
        prediction = predict_student(gender, race, parental_education, lunch, math_score, reading_score, writing_score)
        if prediction == 1:
            st.success("The student will continue in the career")
        else:
            st.error("The student will drop out of college")
