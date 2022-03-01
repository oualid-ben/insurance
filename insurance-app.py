import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Insurance fraud detection: IMAFA

CEO: Gertaldi Negre Aayar

This app predicts if a claim is a fraud or not !

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
        model  = st.sidebar.selectbox('Model',('KNN', 'Random Forest'))
        policy_annual_premium  = st.number_input('policy annual premium ', 433.33, 2047.59, 700.33)
        
        umbrella_limit  = st.number_input('umbrella limit ', -1000000, 10000000, 0)
        capital-gains  = st.number_input('capital gains ', 0, 100500, 700)
        capital-loss  = st.number_input('capital loss ', -111100, 0, -333)
        incident_severity  = st.number_input('incident severity ', 0, 3, 2)
        incident_hour_of_the_day  = st.number_input('incident hour of the day ', 0, 23, 4)
        number_of_vehicles_involved  = st.number_input('number of vehicles involved ', 1,4,2)
        bodily_injuries  = st.number_input('bodily injuries ', 0, 2, 1)
        property_claim  = st.number_input('property claim ', 0, 23670, 1000)

        data = {'model': model,
                'policy_annual_premium': policy_annual_premium,
                
                'umbrella_limit': umbrella_limit,
                'capital-gains': capital-gains,
                'capital-loss': capital-loss,
                'incident_severity': incident_severity,
                'incident_hour_of_the_day': incident_hour_of_the_day,
                'number_of_vehicles_involved': number_of_vehicles_involved,
                'bodily_injuries': bodily_injuries,
                'property_claim': property_claim,
               }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()    

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
insurance_raw = pd.read_csv('https://raw.githubusercontent.com/oualid-ben/data/main/clean_data.csv')
opt = input_df.drop(columns=['model'], axis=1)
df = pd.concat([opt ,insurance_raw],axis=0)

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
