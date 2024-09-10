import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
model_path = 'model/etr_model.pkl'
model = joblib.load(model_path)

# Cargar los escaladores
scaler_X_path = 'model/scaler_train_X.pkl'  
scaler_y_path = 'model/scaler_train_Y.pkl'
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Definir una función para hacer predicciones
def make_prediction(inputs):
    # Convertir las entradas a un DataFrame
    input_df = pd.DataFrame([inputs])
    # Normalizar las entradas usando el mismo escalador que en el entrenamiento
    input_df_normalized = scaler_X.transform(input_df)
    # Hacer la predicción
    prediction_normalized = model.predict(input_df_normalized)
    # Desnormalizar la predicción
    prediction = scaler_y.inverse_transform(prediction_normalized.reshape(-1, 1))
    return prediction[0][0]

# Interfaz de usuario de Streamlit con un cuadro azul marino detrás solo del título
st.markdown("""
    <h1 style="background-color:#001f3f;color:white;padding:10px;border-radius:10px;text-align:center;">
        Aplicación de Predicción de GT_NO2
    </h1>
    """, unsafe_allow_html=True)

# Agregar un espacio antes del primer input
st.markdown("<br>", unsafe_allow_html=True)

# Crear campos de entrada
input1 = st.number_input('Latitud', value=0.0, format="%.6f")
input2 = st.number_input('Longitud', value=0.0, format="%.6f")
input3 = st.number_input('NO2_strat', value=0.0, format="%.6f")
input4 = st.number_input('Presión Tropopáusica', format="%.5f")
input5 = st.number_input('Año', value=0, step=1)
input6 = st.number_input('Mes', value=0, step=1)
input7 = st.number_input('Día', value=0, step=1)

# Botón para hacer la predicción
if st.button('Predecir'):
    inputs = [input1, input2, input3, input4, input5, input6, input7]  # Añadir más entradas si es necesario
    prediction = make_prediction(inputs)
    
    # Formatear la predicción a 4 decimales
    formatted_prediction = f"{prediction:.4f}"
    
    # Mostrar la predicción con fondo verde y centrado en la página
    st.markdown(f"""
        <div style="background-color:#28a745;color:white;padding:20px;border-radius:10px;text-align:center;margin:0 auto;width:50%;">
            <h2><strong>Predicción de GT_NO2: {formatted_prediction}</strong></h2>
        </div>
        """, unsafe_allow_html=True)