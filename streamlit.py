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

loaded_model = load_model('nn1_d.h5')

def office_prediction(input_data):
    ss = load('scaler1.bin')
    le = load('labelEncoder1.bin')

    input_data.iloc[:, 5] = le.transform(input_data.iloc[:, 5])

    input_data = ss.transform(input_data)

    pred = loaded_model.predict(input_data)

    return pred[0,0]

if __name__ == '__main__':
    st.title('Office Ratings Prediction')

    Director_Options = ['Ken Kwapis', 'Ken Whittingham', 'Bryan Gordon', 'Greg Daniels',
       'Amy Heckerling', 'Paul Feig', 'Charles McDougall',
       'Dennie Gordon', 'Victor Nelli Jr.', 'See full summary',
       'Roger Nygard', 'Randall Einhorn', 'Miguel Arteta', 'Tucker Gates',
       'Jeffrey Blitz', 'Harold Ramis', 'Julian Farino', 'Joss Whedon',
       'J.J. Abrams', 'Craig Zisk', 'Paul Lieberstein', 'Jason Reitman',
       'Jennifer Celotta', 'David Rogers', 'Stephen Merchant',
       'Dean Holland', 'Asaad Kelada', 'Gene Stupnitsky', 'Steve Carell',
       'Brent Forrester', 'Lee Eisenberg', 'Reginald Hudlin',
       'Seth Gordon', 'B.J. Novak', 'John Krasinski', 'Marc Webb',
       'Matt Sohn', 'Mindy Kaling', 'Rainn Wilson', 'John Scott',
       'Alex Hardcastle', 'Troy Miller', 'Charlie Grandy', 'Ed Helms',
       'Eric Appel', 'Brian Baumgartner', 'Claire Scanlon', 'Daniel Chun',
       'Bryan Cranston', 'Rodman Flender', 'Kelly Cantley-Kashima',
       'Lee Kirk', 'Jon Favreau', 'Jesse Peretz']

    Season = st.slider('Season', min_value=1, max_value=9)
    Votes = st.number_input('Number of votes', step=100, value=0)
    Viewership = st.number_input('Viewership of the episode', step=0.1, value=0.0)
    Duration = st.number_input('Duration of the episode', step=1, value=0)
    Director = st.selectbox("Director of the episode", options=Director_Options)

    #input_data = np.array([[Season, Votes, Viewership, Duration]])
     #prediction = office_prediction(input_data)
    
    prediction = office_prediction([[Season, Votes, Viewership, Duration, Director]])
    # prediction = round(prediction, 2)

    if st.button('Predict Episode Rating'):
        #st.success(f"Predicted rating of the episode: {prediction}")
        st.success("Prediction: %.1f" % prediction)        