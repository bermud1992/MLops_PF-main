import streamlit as st
import requests

API_URL = "http://fastapi:8000/predict/model_production"

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
        'brokered_by': 8103.0,
        'status': 'sold',
        'city': 'Kennett Square',
        'state': 'Pennsylvania',
        'zip_code': 19348.0,
        'bed': 3.0,
        'bath': 1.0,
        'acre_lot': 1.2,
        'house_size': 1995.0,
        'price': 375900.0,
    }

    # Crear campos de entrada para cada atributo
    input_values = {}
    for key, value in default_values.items():
        input_values[key] = st.text_input(key, value)

    # Convertir valores a los tipos de datos correctos
    string_columns = ['status', 'city', 'state', 'prev_sold_date']
    request_body = {key: [float(value)] if key not in string_columns else [value] for key, value in input_values.items()}

    if st.button("Realizar Predicción"):
        st.write("Realizando predicción...")
        result = predict(request_body)
        st.write("Resultado de la predicción:", result)

if __name__ == "__main__":
    main()
