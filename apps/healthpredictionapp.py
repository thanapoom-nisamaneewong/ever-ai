import streamlit as st
import pandas as pd
import pickle
import numpy as np
from multiapp import MultiApp
from apps import diabetes, hypertension, hyperlipidaemia
def calBMI(bt, bw):
    bt = float(bt) / 100
    bw = float(bw)
    bmi = round(bw / (bt * bt), 2)
    return bmi
def app():
    disease=st.selectbox(' choose a disease',('Diabetes', 'Hypertension', 'Hyperlipidemia'))
    if disease =='Diabetes':
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
            modelFile = 'models/disease_prediction/I_Diabetes_ADB_norm.pkl'
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
    if disease =='Hypertension':
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
        modelFile = 'models/disease_prediction/I_Hypertension_RF_norm.pkl'
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
    if disease =='Hyperlipidemia':
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
        modelFile = 'models/disease_prediction/I_Hyperlipidaemia_RF_norm.pkl'
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
