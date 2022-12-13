import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from joblib import load
import streamlit as st

loaded_model = load_model('nn1.h5')

def office_prediction(input_data):
    ss = load('scaler1.bin')

    input_data = ss.fit_transform(input_data)

    pred = loaded_model.predict(input_data)

    return pred

if __name__ == '__main__':
    st.title('Office Ratings Prediction')

    Season = st.slider('Season', min_value=1, max_value=9)
    Votes = st.number_input('Number of votes', step=10)
    Viewership = st.number_input('Viewership of the episode')
    Duration = st.number_input('Duration of the episode', step=1)

    #input_data = np.array([[Season, Votes, Viewership, Duration]])
     #prediction = office_prediction(input_data)
    
    prediction = office_prediction([[Season, Votes, Viewership, Duration]])

    if st.button('Predict Episode Rating'):
        st.success(f"Predicted rating of the episode: {prediction}")        