# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:19:29 2023

@author: Raghava Varanasi
"""
## Import the libraries
import pandas as pd
import streamlit as st 
from pickle import load
import joblib
from keras.models import load_model
import numpy as np

# Title of the website
st.title('Model Deployment: Machine Fails or Not')

# Choosing between predicting a single observation or an entire dataset
st.subheader('Single Prediction or Dataset Prediction')
predict_option = st.radio('Select One Option:', ('Single Prediction', 'Dataset Prediction'))

# Choosing a regressor
st.subheader('Select the Classification Model')
classifier = st.selectbox('Available Models arranged in descending order of their prediction accuracies: ', ('Decision Tree', 'Random Forest', 'AdaBoost', 'Bagging', 'KNN', 'Stacking', 'Gradient Boost', 'Support Vector - Polynomial Kernel', 'Logistic Regression', 'Support Vector - Linear Kernel'))

# Loading the model based on the user selection
if classifier == 'AdaBoost':
    loaded_model = load(open('AdaBoost.sav', 'rb'))
    st.subheader('AdaBoost Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Decision Tree':
    loaded_model = load(open('Decision_Tree.sav', 'rb'))
    st.subheader('Decision Tree Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Random Forest':
    loaded_model = load(open('Random_Forest.sav', 'rb'))
    st.subheader('Random Forest Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Bagging':
    loaded_model = load(open('Bagging.sav', 'rb'))
    st.subheader('Bagging Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Gradient Boost':
    loaded_model = load(open('Gradient_Boost.sav', 'rb'))
    st.subheader('Gradient Boost Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Stacking':
    loaded_model = load(open('Stacking.sav', 'rb'))
    st.subheader('Stacking Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'KNN':
    loaded_model = load(open('KNN.sav', 'rb'))
    st.subheader('KNN Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Logistic Regression':
    loaded_model = load(open('Logistic_Regression_model.sav', 'rb'))
    st.subheader('Logistic Regression Classification Model')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Support Vector - Polynomial Kernel':
    loaded_model = load(open('SVC_Poly.sav', 'rb'))
    st.subheader('Support Vector Classification Model - Polynomial Kernel')
    st.markdown('Model Score   --   **99.91%**')
elif classifier == 'Support Vector - Linear Kernel':
    loaded_model = load(open('SVC_Linear.sav', 'rb'))
    st.subheader('Support Vector Classification Model - Linear Kernel')
    st.markdown('Model Score   --   **99.91%**')

# Single prediction
if predict_option == 'Single Prediction':
    st.sidebar.header('User Input Parameters - Select values between 0 to 1:')
    
    twf = st.sidebar.radio('TWF:', (0, 1))
    hdf = st.sidebar.radio('HDF:', (0, 1))
    pwf = st.sidebar.radio('PWF:', (0, 1))
    osf = st.sidebar.radio('OSF:', (0, 1))
    
    
    
    data = {'TWF':twf,
            'HDF':hdf,
            'PWF':pwf,
            'OSF':osf}
    df = pd.DataFrame(data,index = [0])
    
    st.subheader('User Input parameters')
    st.write(df)
    
    st.subheader('Make Prediction:')
    if st.button('Predict'):
        st.subheader(f'Predicted Result - {classifier} Classifier')
        result = loaded_model.predict(df)[0]
        #result_prob = np.round((loaded_model.predict_proba(df)[:, 1][0]), 2)

        if result == 1:
            st.markdown('The machine **:red[will fail]**')
        else:
            st.markdown('The machine **:green[will not fail]**')
        

# Dataset prediction
else:
    st.subheader('Upload the dataset')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)

        st.subheader('Make Predictions:')
        if st.button('Predict'):
            prediction = loaded_model.predict(df[['TWF', 'HDF', 'PWF', 'OSF']])
            st.subheader(f'Predicted Result - {classifier} Classifier')

            df['Machine failure prediction'] = prediction
            
            st.write(df)
            
            st.subheader('Download Predictions:')
            st.download_button('Download Predictions', data = df.to_csv().encode('utf-8'), file_name=f'Precited Data - {classifier} Classifier.csv', mime='text/csv')


        












    