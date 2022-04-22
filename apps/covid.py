import streamlit as st
import json
import datetime
import requests
import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from datetime import timedelta
from dateutil import rrule

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout





def get_api_data():
    resp = requests.get('https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-all')
    jdata = resp.json()
    df = pd.json_normalize(jdata)

    return df

def app():
    st.header("""Covid-19 Outbreak Prediction in Thailand""")
    df = get_api_data()

    df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

    df = df[['txn_date', 'new_case']]
    df = df.drop_duplicates()
    df = df.sort_values(by='txn_date')

    dataset_rnn = df[['txn_date', 'new_case']]

    dataset_train, dataset_test = train_test_split(dataset_rnn, test_size=0.25, shuffle=False)

    training_set = dataset_train.iloc[:, 1:2].values

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 20 timesteps and t+1 output
    X_train = []
    y_train = []
    for i in range(20, len(dataset_train)):
        X_train.append(training_set_scaled[i - 20:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Initialize the RNN
    model = keras.Sequential()

    # Add first LSTM layer and Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(None, 1)))
    model.add(Dropout(0.2))

    # Adding second layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding third layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding fourth layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mse')

    # Training the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)


    #
    test_set = dataset_test.iloc[:, 1:2].values

    real_covid_cases = np.concatenate((training_set[0:len(dataset_train)], test_set), axis=0)

    # Getting the predicted number of cases
    scaled_real_covid_cases = sc.fit_transform(real_covid_cases)
    inputs = []

    for i in range(len(dataset_train) + 1, len(real_covid_cases) + 1):
        inputs.append(scaled_real_covid_cases[i - 20:i, 0])

    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

    # Predicting
    predicted_covid_cases = model.predict(inputs)
    predicted_covid_cases = sc.inverse_transform(predicted_covid_cases)
    predicted_covid_cases = predicted_covid_cases.astype('int64')


    # predict the next 30 days case
    sc = MinMaxScaler(feature_range=(0, 1))
    real_covid_cases_scaled = sc.fit_transform(real_covid_cases)
    input_next_pred = real_covid_cases_scaled.reshape(-1, 1)
    list_input = list(input_next_pred)

    lst_output = []
    n_steps = len(real_covid_cases)
    i = 0
    while (i < 30):
        if (len(list_input) > len(real_covid_cases)):
            input_next_pred = np.array(list_input[1:], dtype=object)
            input_next_pred = input_next_pred.reshape(1, -1)
            input_next_pred = input_next_pred.reshape((1, n_steps, 1)).astype(np.float32)
            y_pred = model.predict(input_next_pred)
            y_pred = sc.inverse_transform(y_pred)
            y_pred = y_pred.astype('int64')

            list_input.extend(y_pred[0].tolist())
            list_input = list_input[1:]
            lst_output.extend(y_pred.tolist())
            i = i + 1
        else:
            input_next_pred = input_next_pred.reshape((1, n_steps, 1))
            y_pred = model.predict(input_next_pred)
            y_pred = sc.inverse_transform(y_pred)
            y_pred = y_pred.astype('int64')
            list_input.extend(y_pred[0].tolist())
            lst_output.extend(y_pred.tolist())
            i = i + 1


    new_date = []

    for i in range(1, 31):
        new_date.append(dataset_rnn['txn_date'].max() + timedelta(days=i))

    new_date_df = pd.DataFrame(new_date, columns=['txn_date'])

    first_date = str(new_date_df['txn_date'].min().strftime('%Y/%m/%d'))
    end_date = str(new_date_df['txn_date'].max().strftime('%Y/%m/%d'))

    full_df = pd.DataFrame({'Date': new_date, 'Predicted Cases': lst_output})
    full_df['Predicted Cases'] = full_df['Predicted Cases'].str.get(0)
    full_df['Date'] = full_df['Date'].dt.date

    st.subheader('Predicted Cases in Next 5 Days')
    st.write(full_df.head(5))

    fig = plt.figure(figsize=(20, 10))
    plt.plot(dataset_rnn['txn_date'], real_covid_cases, color='blue', label='Real Covid Cases')
    plt.plot(new_date_df['txn_date'], lst_output, color='red', label='Predicted Covid Cases')
    plt.title('Covid-19 Forecasting Trend in Next 30 Days from ' + first_date + ' to ' + end_date)
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.legend()
    st.pyplot(fig)

