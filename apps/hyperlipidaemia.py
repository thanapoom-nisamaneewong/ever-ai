import streamlit as st
import pandas as pd
import numpy as np
import pickle

def app():
    st.write('## Hyperlipidaemia Disease Prediction')

    st.sidebar.header('User Input Features')

    st.sidebar.markdown('[Example CSV input file]')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            age = st.sidebar.slider('Age', 0, 120, 60)
            sex = st.sidebar.selectbox('Sex', ('male', 'female'))
            bps = st.sidebar.slider('BPS', 30.0, 180.0, 75.0)
            bpd = st.sidebar.slider('BPD', 30.0, 180.0, 75.0)
            bw = st.sidebar.slider('Weight', 30.0, 180.0, 75.0)
            bt = st.sidebar.slider('Height', 30.0, 180.0, 75.0)
            waist = st.sidebar.slider('Waist', 30.0, 180.0, 75.0)

            diabetes_disease = st.sidebar.selectbox('Diabetes', ('yes', 'no'))
            hypertension_disease = st.sidebar.selectbox('Hypertension', ('yes', 'no'))
            ischaemi_heart_disease = st.sidebar.selectbox('Ischaemi Heart Disease', ('yes', 'no'))
            chronic_kidney_disease = st.sidebar.selectbox('Chronic Kidney Disease', ('yes', 'no'))
            kidney_stones_disease = st.sidebar.selectbox('kidney Stones Disease', ('yes', 'no'))
            stroke_disease = st.sidebar.selectbox('Stroke Disease', ('yes', 'no'))
            fatty_liver_disease = st.sidebar.selectbox('Fatty Liver', ('yes', 'no'))
            HDL_chelesterol = st.sidebar.slider('HDL_chelesterol', 0.0, 1000.0, 200.0)
            LDL_chelesterol = st.sidebar.slider('LDL_chelesterol', 0.0, 1000.0, 200.0)
            triglyceride = st.sidebar.slider('Triglyceride', 0.0, 1000.0, 200.0)
            creatinine = st.sidebar.slider('Creatinine', 0.0, 1000.0, 200.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 120.0, 12.0)
            cholesterol = st.sidebar.slider('cholesterol', 0.0, 1000.0, 200.0)
            total_bilrubin = st.sidebar.slider('Total Bilrubin', 0.0, 1000.0, 200.0)

            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'patientAge': age,
                    'patientSexName': sex,
                    'vital_bpd': bpd,
                    'vital_bps': bps,
                    'vital_bmi': bmi,
                    'vital_waist': waist,
                    'fatty_liver_disease': fatty_liver_disease,
                    'kidney_stones_disease': kidney_stones_disease,
                    'diabetes_disease': diabetes_disease,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'ischaemi_heart_disease': ischaemi_heart_disease,
                    'hypertension_disease': hypertension_disease,
                    'stroke_disease': stroke_disease,
                    '_103': triglyceride,
                    '_15': HDL_chelesterol,
                    '_78': creatinine,
                    '_18': LDL_chelesterol,
                    '_1021': cholesterol,
                    '_32': total_bilrubin,
                    '_452': HbA1c
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

    disease = ['fatty_liver_disease', 'kidney_stones_disease', 'diabetes_disease', 'chronic_kidney_disease',
               'ischaemi_heart_disease', 'hypertension_disease', 'stroke_disease']
    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    # Normalize
    scalerFile = 'scalers/I_Hyperlipidaemia__scaler.sav'
    scalerLoad = pickle.load(open(scalerFile, 'rb'))

    # get pickle model
    modelFile = 'models/I_Hyperlipidaemia_RF_norm.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    df_norm = scalerLoad.transform(input_df)
    predict = model.predict(df_norm)[0]
    probability = model.predict_proba(df_norm)

    # Translate Result
    if predict == 1:
        prob = round(probability[:, 1][0] * 100, 2)
        result = 'Your Hyperlipidemia Risk is ' + str(prob) + ' %'
    else:
        prob = round(probability[:, 0][0] * 100, 2)
        result = 'Your Hyperlipidemia Risk is ' + str(prob) + ' %'

    st.subheader('Prediction')
    st.write(result)