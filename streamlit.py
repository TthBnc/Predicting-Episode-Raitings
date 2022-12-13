import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import streamlit as st

loaded_model = load_model('nn1.h5')

def office_prediction(input_data):
    ss = StandardScaler()

    input_data = ss.fit_transform(input_data)

    pred = loaded_model.predict(input_data)

    return pred

if __name__ == '__main__':
    st.title('Office Ratings Prediction')

    Season = 2
    Votes = 2000
    Viewership = 2.0
    Duration = 23

    prediction = 0

    if st.button('Predict Episode Rating'):
        prediction = office_prediction([[Season, Votes, Viewership, Duration]])

        st.success(f"Predicted rating of the episode: {prediction}")        