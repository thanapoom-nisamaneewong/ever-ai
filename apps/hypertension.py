import streamlit as st
import pandas as pd
import numpy as np
import pickle

def calBMI(bt, bw):
    bt = float(bt) / 100
    bw = float(bw)
    bmi = round(bw / (bt * bt), 2)
    return bmi

def app():
    st.write('## Hypertension Disease Prediction')

    st.sidebar.header('User Input Features')

    st.sidebar.markdown('[Example CSV input file]')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        #input_df['vital_bmi'] = input_df.apply(lambda x: calBMI(x.vital_bt, x.vital_bw), axis=1)

    else:
        def user_input_features():
            age = st.sidebar.slider('Age', 0, 120, 60)
            sex = st.sidebar.selectbox('Sex', ('male', 'female'))
            bps = st.sidebar.slider('BPS', 30.0, 180.0, 75.0)
            bpd = st.sidebar.slider('BPD', 30.0, 180.0, 75.0)
            bw = st.sidebar.slider('Weight', 30.0, 180.0, 75.0)
            bt = st.sidebar.slider('Height', 30.0, 180.0, 75.0)
            hyperlipidaemia_disease = st.sidebar.selectbox('Hyperlipidaemia', ('yes', 'no'))
            stroke_disease = st.sidebar.selectbox('Stroke', ('yes', 'no'))
            chronic_kidney_disease = st.sidebar.selectbox('chronic_kidney_disease', ('yes', 'no'))
            chronic_obstructive_pulmonary_disease = st.sidebar.selectbox('chronic_obstructive_pulmonary_disease', ('yes', 'no'))
            fasting_glucose = st.sidebar.slider('Fasting Glucose', 0.0, 1000.0, 200.0)
            LDL_chelesterol = st.sidebar.slider('LDL_chelesterol', 0.0, 1000.0, 200.0)
            HDL_cholesterol = st.sidebar.slider('HDL_chelesterol', 0.0, 1000.0, 200.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 120.0, 12.0)
            micro_albumin_uria = st.sidebar.slider('Micro Albumin Uria', 0.0, 1000.0, 200.0)


            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'patientSexName': sex,
                    'patientAge': age,
                    'vital_bpd': bpd,
                    'vital_bps': bps,
                    'vital_bmi': bmi,
                    'hyperlipidaemia_disease': hyperlipidaemia_disease,
                    'stroke_disease': stroke_disease,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'chronic_obstructive_pulmonary_disease': chronic_obstructive_pulmonary_disease,
                    '_550': micro_albumin_uria,
                    '_452': HbA1c,
                    '_15': HDL_cholesterol,
                    '_18': LDL_chelesterol,
                    '_7': fasting_glucose
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        data = user_input_features()

    # Displays the user input features
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(data)
    else:
        st.write('Awaiting CSV file to be uploaded.')
        st.write(data)

    input_df = pd.DataFrame(data, index=[0])

    input_df['patientSexName'].replace(['male', 'female'], [1, 0], inplace=True)

    disease = ['hyperlipidaemia_disease', 'stroke_disease', 'chronic_kidney_disease',
               'chronic_obstructive_pulmonary_disease']
    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    # Normalize
    scalerFile = 'scalers/I_Hypertension__scaler.sav'
    scalerLoad = pickle.load(open(scalerFile, 'rb'))

    # get pickle model
    modelFile = 'models/I_Hypertension_GBC_norm.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    df_norm = scalerLoad.transform(input_df)
    predict = model.predict(df_norm)[0]
    probability = model.predict_proba(df_norm)

    # Translate Result
    if predict == 1:
        prob = round(probability[:, 1][0] * 100, 2)
        result = 'Your Hypertension Risk is ' + str(prob) + ' %'
    else:
        prob = round(probability[:, 0][0] * 100, 2)
        result = 'Your Hypertension Risk is ' + str(prob) + ' %'

    st.subheader('Prediction')
    st.write(result)