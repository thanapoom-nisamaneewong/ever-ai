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
        input_df = pd.read_csv(uploaded_file)
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
            acute_renal_failure_disease = st.sidebar.selectbox('Acute Renal Failure Disease', ('yes', 'no'))
            chronic_kidney_disease = st.sidebar.selectbox('Chronic Kidney Disease', ('yes', 'no'))
            kidney_stones_disease = st.sidebar.selectbox('kidney Stones Disease', ('yes', 'no'))
            cholesterol = st.sidebar.slider('cholesterol', 0.0, 1000.0, 200.0)
            fasting_glucose = st.sidebar.slider('Fasting Glucose', 0.0, 1000.0, 200.0)
            creatinine = st.sidebar.slider('Creatinine', 0.0, 1000.0, 200.0)
            triglyceride = st.sidebar.slider('Triglyceride', 0.0, 1000.0, 200.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 120.0, 12.0)
            HGB = st.sidebar.slider('HGB', 0.0, 1000.0, 200.0)
            alkaline_phosphatase = st.sidebar.slider('Alkaline Phosphatase', 0.0, 1000.0, 200.0)
            albumin = st.sidebar.slider('Albumin', 0.0, 1000.0, 200.0)
            globulin = st.sidebar.slider('Globulin', 0.0, 120.0, 12.0)
            total_bilrubin = st.sidebar.slider('Total Bilrubin', 0.0, 1000.0, 200.0)
            total_protein = st.sidebar.slider('Total Protein', 0.0, 1000.0, 200.0)
            direct_bilrubin = st.sidebar.slider('Direct Bilrubin', 0.0, 1000.0, 200.0)
            uric_acid = st.sidebar.slider('Uric acid', 0.0, 1000.0, 200.0)
            MCV = st.sidebar.slider('MCV', 0.0, 1000.0, 200.0)
            MCH = st.sidebar.slider('MCH', 0.0, 1000.0, 200.0)

            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'patientAge': age,
                    'patientSexName': sex,
                    'vital_bpd': bpd,
                    'vital_bps': bps,
                    'vital_bmi': bmi,
                    'vital_waist': waist,
                    'diabetes_disease': diabetes_disease,
                    'hypertension_disease': hypertension_disease,
                    'ischaemi_heart_disease': ischaemi_heart_disease,
                    'acute_renal_failure_disease': acute_renal_failure_disease,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'kidney_stones_disease': kidney_stones_disease,
                    '_102': cholesterol,
                    '_7': fasting_glucose,
                    '_78': creatinine,
                    '_103': triglyceride,
                    '_7952': HbA1c,
                    '_346': HGB,
                    '_5086': alkaline_phosphatase,
                    '_193': albumin,
                    '_5081': globulin,
                    '_32': total_bilrubin,
                    '_94': total_protein,
                    '_36': direct_bilrubin,
                    '_5074': uric_acid,
                    '_5014': MCV,
                    '_5015': MCH
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

    disease = ['diabetes_disease', 'hypertension_disease', 'ischaemi_heart_disease',
               'acute_renal_failure_disease', 'chronic_kidney_disease', 'kidney_stones_disease']
    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    # Normalize
    scalerFile = 'scalers/C_Hyperlipidaemia_scaler.sav'
    scalerLoad = pickle.load(open(scalerFile, 'rb'))

    df_norm = scalerLoad.transform(input_df)

    # get pickle model
    modelFile = 'models/C_Hyperlipidaemia_Model_LR.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    predict = model.predict(df_norm)[0]
    probability = model.predict_proba(df_norm)

    # Translate Result
    if predict == 1:
        prob = str(round(probability[:, 1][0] * 100, 2))
        result = 'Your Hyperlipidemia risk is at ' + prob + '%.'
    else:
        prob = str(round(probability[:, 0][0] * 100, 2))
        result = 'Your Hyperlipidemia risk is at ' + prob + '%.'

    st.subheader('Prediction')
    st.write(result)