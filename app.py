import os, pickle
import streamlit as st
import pandas as pd

filename = 'perceptron_model.pkl'
with open(os.path.join('models', filename), 'rb') as f:
    perceptron, variables = pickle.load(f)

data = st.file_uploader('Introduzca el set de deployment (csv format)', type='csv')
if data is not None:
    data = pd.read_csv(data)
    #validating data
    validation = set(variables)-set(data.columns)
    if len(validation):
        st.error(f'columns not found in data {", ".join(validation)}', icon="🚨")
        st.stop()


X = data[variables.tolist()]
predictions = perceptron.predict(X)
predictions = ['Abandona' if pred==1 else 'Permanece' for pred in predictions]
df = X.assign(Predicción=predictions)
df