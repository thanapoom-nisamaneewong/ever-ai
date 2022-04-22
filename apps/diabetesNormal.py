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
    st.write('## Diabetes Disease Prediction')

    st.sidebar.header('User Input Features')

    st.sidebar.markdown('[Example CSV input file](https://drive.google.com/file/d/116dJotQbAa-X8Iu07GNLGj399CYWYoZS/view?usp=sharing)')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df['vital_bmi'] = input_df.apply(lambda x: calBMI(x.vital_bt, x.vital_bw), axis=1)
        input_df = input_df[['patientSexName', 'patientAge', 'vital_bpd', 'vital_bps', 'vital_bmi',
                            'vital_waist', 'hypertension_disease', 'hyperlipidaemia_disease', 'pregnancy_disease',
                            'fasting_glucose', 'creatinine', 'sodium', 'potassium', 'HDL_chelesterol',
                             'FBS', 'LDL_chelesterol', 'HbA1c']]
    else:
        def user_input_features():
            age = st.sidebar.slider('Age', 0, 120, 60)
            sex = st.sidebar.selectbox('Sex', ('male', 'female'))
            bps = st.sidebar.slider('BPS', 30.0, 180.0, 75.0)
            bpd = st.sidebar.slider('BPD', 30.0, 180.0, 75.0)
            bw = st.sidebar.slider('Weight', 30.0, 180.0, 75.0)
            bt = st.sidebar.slider('Height', 30.0, 180.0, 75.0)
            waist = st.sidebar.slider('Waist', 30.0, 180.0, 75.0)
            hypertension_disease = st.sidebar.selectbox('Hypertension', ('yes', 'no'))
            hyperlipidaemia_disease = st.sidebar.selectbox('Hyperlipidaemia', ('yes', 'no'))
            pregnancy_disease = st.sidebar.selectbox('Pregnancy', ('yes', 'no'))
            fasting_glucose = st.sidebar.slider('Fasting Glucose', 0.0, 1000.0, 200.0)
            creatinine = st.sidebar.slider('Creatinine', 0.0, 1000.0, 200.0)
            sodium = st.sidebar.slider('Sodium', 0.0, 1000.0, 200.0)
            potassium = st.sidebar.slider('Potassium', 0.0, 1000.0, 200.0)
            HDL_chelesterol = st.sidebar.slider('HDL_chelesterol', 0.0, 1000.0, 200.0)
            FBS = st.sidebar.slider('FBS', 0.0, 120.0, 12.0)
            LDL_chelesterol = st.sidebar.slider('LDL_chelesterol', 0.0, 1000.0, 200.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 1000.0, 200.0)

            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'patientSexName': sex,
                    'patientAge': age,
                    'vital_bpd': bpd,
                    'vital_bps': bps,
                    'vital_bmi': bmi,
                    'vital_waist': waist,
                    'hypertension_disease': hypertension_disease,
                    'hyperlipidaemia_disease': hyperlipidaemia_disease,
                    'pregnancy_disease': pregnancy_disease,
                    '_7': fasting_glucose,
                    '_78': creatinine,
                    '_5093': sodium,
                    '_5094': potassium,
                    '_15': HDL_chelesterol,
                    '_76': FBS,
                    '_18': LDL_chelesterol,
                    '_452': HbA1c
                    }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    # Displays the user input features
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(input_df)
    else:
        st.write('Awaiting CSV file to be uploaded.')
        st.write(input_df)

    input_df['patientSexName'].replace(['male', 'female'], [1, 0], inplace=True)

    disease = ['hypertension_disease', 'hyperlipidaemia_disease', 'pregnancy_disease']
    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    # get pickle model
    modelFile = 'models/C_Diabetes_Model_LR_normal.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    predict = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)

    # Translate Result
    if predict == 1:
        prob = str(round(probability[:, 1][0] * 100, 2))
        result = 'Your Diabetes risk is at ' + prob + '%.'
    else:
        prob = str(round(probability[:, 0][0] * 100, 2))
        result = 'Your Diabetes risk is at ' + prob + '%.'

    st.subheader('Prediction')
    st.write(result)