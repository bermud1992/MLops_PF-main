import streamlit as st
import requests

API_URL = "http://10.43.101.151:8088/predict/DecisionTreeClassifier"

def predict(request_body):
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_URL, json=request_body, headers=headers
    )
    return response.json()

def main():
    st.title("Aplicación de Streamlit para API POST")

    # Definir los valores por defecto basados en el esquema
    default_values = {       
        'race': 'Caucasian',
        'gender': 'Female',
        'age': '[80-90)',
        'admission_type_id': 1,
        'discharge_disposition_id': 1,
        'admission_source_id': 7,
        'metformin': 'Steady',
        'repaglinide': 'No',
        'nateglinide': 'No',
        'chlorpropamide': 'No',
        'glimepiride': 'No',
        'acetohexamide': 'No',
        'glipizide': 'No',
        'glyburide': 'Steady',
        'tolbutamide': 'No',
        'pioglitazone': 'No',
        'rosiglitazone': 'No',
        'acarbose': 'No',
        'miglitol': 'No',
        'troglitazone': 'No',
        'tolazamide': 'No',
        'insulin': 'No',
        'glyburide_metformin': 'No',
        'glipizide_metformin': 'No',
        'metformin_rosiglitazone': 'No',
        'metformin_pioglitazone': 'No',
        'change': 'Ch',
        'diabetesMed': 'Yes',
        'diag_1': 435,
        'diag_2': 491,
        'diag_3': 428,
        'time_in_hospital': 10,
        'num_lab_procedures': 45,
        'num_procedures': 0,
        'num_medications': 14,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 0,
        'number_diagnoses': 9
    }

    # Crear campos de entrada para cada atributo
    input_values = {}
    for key, value in default_values.items():
        input_values[key] = st.text_input(key, value)

    # Convertir valores a los tipos de datos correctos
    request_body = {key: [int(value)] if key not in ['race', 'gender', 'age', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide'
        ,'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide_metformin'
        ,'glipizide_metformin', 'metformin_rosiglitazone','metformin_pioglitazone', 'change', 'diabetesMed'] else [value] for key, value in input_values.items()}

    if st.button("Realizar Predicción"):
        st.write("Realizando predicción...")
        result = predict(request_body)
        st.write("Resultado de la predicción:", result)

if __name__ == "__main__":
    main()
