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

    st.sidebar.markdown('[Example CSV input file](https://drive.google.com/file/d/1DZFvMhSxHHkNzyz-DTzrN23AxFbcvhdn/view?usp=sharing)')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df['vital_bmi'] = input_df.apply(lambda x: calBMI(x.vital_bt, x.vital_bw), axis=1)
        input_df = input_df[['patientSexName', 'patientAge', 'vital_bpd', 'vital_bps', 'vital_bw', 'vital_bmi', 'vital_waist',
                             'diabetes_disease', 'hyperlipidaemia_disease', 'stroke_disease', 'rheumatoid_arthritis_disease',
                             'chronic_kidney_disease', 'chronic_obstructive_pulmonary_disease', 'creatinine', 'HbA1c',
                             'micro_albumin_uria', 'albumin_uria', 'triglyceride', 'cholesterol', 'urin_acid',
                             'LDL_chelesterol', 'HDL_chelesterol']]
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
            hyperlipidaemia_disease = st.sidebar.selectbox('Hyperlipidaemia', ('yes', 'no'))
            stroke_disease = st.sidebar.selectbox('Stroke', ('yes', 'no'))
            rheumatoid_arthritis_disease = st.sidebar.selectbox('rheumatoid_arthritis_disease', ('yes', 'no'))
            chronic_kidney_disease = st.sidebar.selectbox('chronic_kidney_disease', ('yes', 'no'))
            chronic_obstructive_pulmonary_disease = st.sidebar.selectbox('chronic_obstructive_pulmonary_disease', ('yes', 'no'))
            creatinine = st.sidebar.slider('creatinine', 0.0, 1000.0, 200.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 120.0, 12.0)
            micro_albumin_uria = st.sidebar.slider('Micro Albumin Uria', 0.0, 1000.0, 200.0)
            albumin_uria = st.sidebar.slider('Albumin', 0.0, 1000.0, 200.0)
            triglyceride = st.sidebar.slider('Triglyceride', 0.0, 1000.0, 200.0)
            cholesterol = st.sidebar.slider('cholesterol', 0.0, 1000.0, 200.0)
            urin_acid = st.sidebar.slider('Uric Acid', 0.0, 1000.0, 200.0)
            LDL_chelesterol = st.sidebar.slider('LDL_chelesterol', 0.0, 1000.0, 200.0)
            HDL_chelesterol = st.sidebar.slider('HDL_chelesterol', 0.0, 1000.0, 200.0)

            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'patientAge': age,
                    'patientSexName': sex,
                    'vital_bpd': bpd,
                    'vital_bps': bps,
                    'vital_bw': bw,
                    'vital_bmi': bmi,
                    'vital_waist': waist,
                    'diabetes_disease': diabetes_disease,
                    'hyperlipidaemia_disease': hyperlipidaemia_disease,
                    'stroke_disease': stroke_disease,
                    'rheumatoid_arthritis_disease': rheumatoid_arthritis_disease ,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'chronic_obstructive_pulmonary_disease': chronic_obstructive_pulmonary_disease,
                    '_5073': creatinine,
                    '_452': HbA1c,
                    '_550': micro_albumin_uria,
                    '_379': albumin_uria,
                    '_5076': triglyceride,
                    '_102': cholesterol,
                    '_79': urin_acid,
                    '_18': LDL_chelesterol,
                    '_15': HDL_chelesterol
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

    disease = ['diabetes_disease', 'hyperlipidaemia_disease', 'stroke_disease', 'rheumatoid_arthritis_disease',
                'chronic_kidney_disease', 'chronic_obstructive_pulmonary_disease',]

    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    # Normalize
    scalerFile = 'scalers/C_Hypertension_scaler.sav'
    scalerLoad = pickle.load(open(scalerFile, 'rb'))

    df_norm = scalerLoad.transform(input_df)

    # get pickle model
    modelFile = 'models/C_Hypertension_Model_GNB.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    predict = model.predict(df_norm)[0]
    probability = model.predict_proba(df_norm)

    # Translate Result
    if predict == 1:
        prob = str(round(probability[:, 1][0] * 100, 2))
        result = 'Your Hypertension risk is at ' + prob + '%.'
    else:
        prob = str(round(probability[:, 0][0] * 100, 2))
        result = 'Your Hypertension risk is at ' + prob + '%.'

    st.subheader('Prediction')
    st.write(result)