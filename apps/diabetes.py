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

    st.sidebar.markdown('Example CSV input file: [link](https://drive.google.com/file/d/15Wv7iJsaHPn7OwiIfvyShj7TUwnKWamB/view?usp=sharing)')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    else:
        def user_input_features():
            age = st.sidebar.slider('Age', 0, 100, 65)
            sex = st.sidebar.selectbox('Sex', ('male', 'female'))
            bps = st.sidebar.slider('BPS', 30.0, 180.0, 75.0)
            bpd = st.sidebar.slider('BPD', 30.0, 180.0, 75.0)
            bw = st.sidebar.slider('Weight', 30.0, 200.0, 45.0)
            bt = st.sidebar.slider('Height', 30.0, 200.0, 150.0)
            waist = st.sidebar.slider('Waist', 30.0, 150.0, 90.0)
            hypertension_disease = st.sidebar.selectbox('Hypertension', ('yes', 'no'))
            hyperlipidaemia_disease = st.sidebar.selectbox('Hyperlipidaemia', ('yes', 'no'))
            ischaemi_heart_disease = st.sidebar.selectbox('Ischaemi Heart Disease', ('yes', 'no'))
            chronic_kidney_disease = st.sidebar.selectbox('CKD', ('yes', 'no'))
            gout_disease = st.sidebar.selectbox('Gout', ('yes', 'no'))
            fasting_glucose = st.sidebar.slider('Fasting Glucose', 0.0, 1000.0, 200.0)
            creatinine = st.sidebar.slider('Creatinine', 0.0, 30.0, 1.0)
            potassium = st.sidebar.slider('Potassium', 0.0, 7.0, 3.0)
            HDL_chelesterol = st.sidebar.slider('HDL_chelesterol', 0.0, 100.0, 50.0)
            LDL_chelesterol = st.sidebar.slider('LDL_chelesterol', 0.0, 1000.0, 500.0)
            HbA1c = st.sidebar.slider('HbA1c', 0.0, 20.0, 5.0)
            eGFR = st.sidebar.slider('eGFR', 0.0, 15.0, 8.0)
            CO2 = st.sidebar.slider('CO2', 0.0, 1000.0, 200.0)
            Cholesterol = st.sidebar.slider('Cholesterol', 0.0, 1000.0, 200.0)
            BUN = st.sidebar.slider('BUN', 0.0, 100.0, 10.0)

            bt = float(bt) / 100
            bw = float(bw)
            bmi = round(bw / (bt * bt), 2)

            data = {'Sex': sex,
                    'Age': age,
                    'BPD': bpd,
                    'BPS': bps,
                    'BMI': bmi,
                    'WAIST': waist,
                    'hypertension_disease': hypertension_disease,
                    'hyperlipidaemia_disease': hyperlipidaemia_disease,
                    'fasting_glucose': fasting_glucose,
                    'creatinine': creatinine,
                    'potassium': potassium,
                    'HDL_chelesterol': HDL_chelesterol,
                    'LDL_chelesterol': LDL_chelesterol,
                    'HbA1c': HbA1c,
                    'chronic_kidney_disease': chronic_kidney_disease,
                    'gout_disease': gout_disease,
                    'ischaemi_heart_disease': ischaemi_heart_disease,
                    'eGFR': eGFR,
                    'CO2': CO2,
                    'Cholesterol': Cholesterol,
                    'BUN': BUN}

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

    input_df = pd.DataFrame(data)

    input_df['Sex'].replace(dict.fromkeys(['ชาย', 'ช', 'male', 'Male', 'M', 'm'], '1'), inplace=True)
    input_df['Sex'].replace(dict.fromkeys(['หญิง', 'ญ', 'female', 'Female', 'F', 'f'], '0'), inplace=True)

    disease = ['hypertension_disease', 'hyperlipidaemia_disease', 'chronic_kidney_disease', 'gout_disease', 'ischaemi_heart_disease']
    for d in disease:
        input_df[d].replace(['yes', 'no'], [1, 0], inplace=True)

    lab = ['fasting_glucose', 'creatinine', 'potassium', 'HDL_chelesterol', 'LDL_chelesterol', 'HbA1c', 'eGFR', 'CO2', 'Cholesterol', 'BUN']
    for col in lab:
        input_df[col].replace(np.nan, 0, inplace=True)
        input_df[col] = input_df[col].apply(lambda x: 0 if x == '-' or x == '' or x == np.nan else x)

    # Normalize
    scalerFile = 'scalers/I_Diabetes__scaler.sav'
    scalerLoad = pickle.load(open(scalerFile, 'rb'))

    # get pickle model
    modelFile = 'models/I_Diabetes_ADB_norm.pkl'
    model = pickle.load(open(modelFile, 'rb'))

    # Predict by model
    df_norm = scalerLoad.transform(input_df)
    predict = model.predict(df_norm)

    probability = model.predict_proba(df_norm)

    pro_1 = probability[:, 1]
    pro_0 = probability[:, 0]

    input_df['resultRisk'] = pro_1
    #input_df['resultUnrisk'] = pro_0

    input_df['resultRisk'] = round(input_df['resultRisk'] * 100, 2)
    #input_df['resultUnrisk'] = round(input_df['resultUnrisk'] * 100, 2)

    #input_df['Predicted'] = predict
    #input_df['PredictedResult'] = input_df['Predicted'].replace([1, 0], [input_df['resultRisk'], input_df['resultUnrisk']])
    input_df['Predicted_Risk_Prob'] = input_df['resultRisk']

    st.subheader('Prediction')
    st.write(input_df['Predicted_Risk_Prob'])
