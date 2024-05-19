# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np 
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
from pickle import load

st.title('Prediction on Global Development Measurement')

st.sidebar.header('User Input Parameters')

def user_input_features():
    birth_rate = st.sidebar.number_input("Birth Rate")
    co2_emission = st.sidebar.number_input('CO2 Emissions',min_value=0.000 )
    days_to_start_business = st.sidebar.number_input('No. of Days to Start Business',max_value=80.00)
    energy_usage = st.sidebar.number_input('Energy Usage',min_value=0.000)
    GDP = st.sidebar.number_input('GDP ($)',min_value=0.000)
    health_exp_percent_GDP = st.sidebar.number_input('Health Exp % GDP value')
    health_exp_percapita = st.sidebar.number_input('Health Exp/Capita',min_value=0.000)
    infant_mortality_rate = st.sidebar.number_input('Infant Mortality Rate')
    internet_usage = st.sidebar.number_input('Internet Usage duration')
    lending_interest = st.sidebar.number_input('Lending Interest rate')
    life_expectancy_female = st.sidebar.number_input('Life Expectancy Female')
    life_expectancy_male = st.sidebar.number_input('Life Expectancy Male')
    mobile_phone_usage = st.sidebar.number_input('Mobile Phone Usage duration')
    population_0_14 =  st.sidebar.number_input('Population between 0-14 Years')
    population_15_64 =  st.sidebar.number_input('Population between 15-64 Years')
    population_65_plus =  st.sidebar.number_input('Population between 65+ Years')
    population_total = st.sidebar.number_input('Population Total',min_value=0.000)
    population_urban = st.sidebar.number_input('Population Urban')
    tourism_inbound = st.sidebar.number_input('Tourism Inbound ($)',min_value=0.000)
    tourism_outbound = st.sidebar.number_input('Tourism Outbound ($)',min_value=0.000)
    country = st.sidebar.text_input('Enter Country')

    data = {'Birth Rate' : birth_rate, 'CO2 Emissions': co2_emission,
       'Days to Start Business' : days_to_start_business, 'Energy Usage' : energy_usage,
       'GDP':GDP, 'Health Exp % GDP': health_exp_percent_GDP,'Health Exp/Capita': health_exp_percapita,
       'Infant Mortality Rate' : infant_mortality_rate, 'Internet Usage':internet_usage,
       'Lending Interest': lending_interest, 'Life Expectancy Female':life_expectancy_female, 'Life Expectancy Male':life_expectancy_male,
       'Mobile Phone Usage' : mobile_phone_usage, 'Population 0-14' : population_0_14 , 'Population 15-64' : population_15_64 ,
       'Population 65+' : population_65_plus , 'Population Total' : population_total , 
       'Population Urban' : population_urban, 'Tourism Inbound' : tourism_inbound,'Tourism Outbound': tourism_outbound,'Country': country}
    
    features = pd.DataFrame(data,index = [0])
    features = features.drop(columns=['Country'], axis=1)
    return features 
df = user_input_features()
st.subheader('User Input parameters')
st.write(df) 

# load the model from disk
loaded_model = load(open('Global_Dev.sav', 'rb'))
prediction = loaded_model.predict(df)
st.subheader('Predicted Result')


def analysis(prediction):
    if prediction == 0:
        return 'Developing Country'
    elif prediction == 1:
        return 'Under Developed Country'
    elif prediction == 2:
        return 'Developed Country'
    else:
        return 'Small Country'

# st.write(analysis(prediction))

if st.button("Predict"):
    st.write(analysis(prediction))