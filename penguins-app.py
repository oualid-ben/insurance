import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Insurance fraud detection: IMAFA

This app predicts if a claim is a fraud or not

Dataset [ins_cl_dataset](https://github.com/oualid-ben/data/blob/main/ins_cl_dataset.csv).
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/oualid-ben/data/main/clean_data_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        model  = st.sidebar.selectbox('Model',('KNN','Tree decision'))
        number_of_vehicles_involved = st.sidebar.slider('number of vehicles involved', 1,3,2)
        witnesses = st.sidebar.slider('Witnesses', 0,3,1)
        data = {'model': Model,
                'number_of_vehicles_involved': number_of_vehicles_involved,
                'witnesses': witnesses,
               }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
insurance_raw = pd.read_csv('https://raw.githubusercontent.com/oualid-ben/data/main/clean_data.csv')
opt = input_df.drop(columns=['Model'], axis=1)
df = pd.concat([input_df,opt],axis=0)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('insurance.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
insurance_species = np.array(['Yes','No'])
st.write(insurance_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
