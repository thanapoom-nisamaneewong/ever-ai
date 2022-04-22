import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow
from tensorflow.keras.models import Model, load_model
def app():

    st.header("""Drug Recommendation System""")

    st.text("""List of 20 medicines: AMLODIPINE, ANALGESICBALM, ASPIRIN, ATENOLOL, ATORVASTATIN, 
    CALCIUMCARBONATE, DOXAZOSIN, ENALAPRIL, FUROSEMIDE, GLIPIZIDE, HYDROCHLOROTHIAZIDE, 
    LORAZEPAM, LOSARTAN, METFORMIN, METHYLSALICYLATE+MENTHOL+CAMPLOR, OMEPRAZOLE, 
    PARACETAMOL, PIOGLITAZONE, SIMVASTATIN, VITAMINBCOMPLEX.""")

    st.text("")

    st.sidebar.header('User Input Features')


    # Collects user input features into dataframe

    age = st.sidebar.slider('Age', 0, 120, 60)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bps = st.sidebar.slider('BPS', 30.0, 300.0, 120.0)
    bpd = st.sidebar.slider('BPD', 30.0, 180.0, 75.0)
    bw = st.sidebar.slider('Weight', 35.0, 150.0, 75.0)
    bt = st.sidebar.slider('Height', 100.0, 200.0, 150.0)
    pulse = st.sidebar.slider('Pulse', 30.0, 200.0, 80.0)
    temperature = st.sidebar.slider('Temperature', 35.0, 45.0, 36.0)
    waist = st.sidebar.slider('Waist', 40.0, 200.0, 70.0)
    fasting_glucose = st.sidebar.slider('Fasting Glucose', 0.0, 1000.0, 200.0)
    creatinine = st.sidebar.slider('Creatinine', 0.0, 100.0, 20.0)
    HbA1c = st.sidebar.slider('HbA1c', 0.0, 200.0, 20.0)
    eGFR = st.sidebar.slider('eGFR', 0.0, 200.0, 20.0)
    bun = st.sidebar.slider('BUN', 0.0, 200.0, 20.0)
    LDL_chelesterol = st.sidebar.slider('LDL Cholesterol', 0.0, 500.0, 200.0)
    HDL_chelesterol = st.sidebar.slider('HDL Cholesterol', 0.0, 500.0, 200.0)
    potassium = st.sidebar.slider('Potassium', 0.0, 10.0, 0.0)
    sodium = st.sidebar.slider('Sodium', 0.0, 200.0, 20.0)
    chloride = st.sidebar.slider('Chloride', 0.0, 200.0, 20.0)
    CO2 = st.sidebar.slider('CO2', 0.0, 100.0, 20.0)


    bt = float(bt) / 100
    bw = float(bw)
    bmi = round(bw / (bt * bt), 2)

    if sex == 'male':
        sex = 1
    else:
        sex = 0

    data = {'patientSexName': sex,
                'patientAge': age,
                'vital_bpd': bpd,
                'vital_bps': bps,
                'vital_pulse': pulse,
                'vital_temperature': temperature,
                'vital_bmi': bmi,
                'vital_waist': waist,
                'Fasting_glucose': fasting_glucose,
                'Creatinine': creatinine,
                'HbA1c': HbA1c,
                'eGFR': eGFR,
                'BUN': bun,
                'LDL_chelesterol': LDL_chelesterol,
                'HDL_cholesterol': HDL_chelesterol,
                'Potassium': potassium,
                'Sodium': sodium,
                'Chloride': chloride,
                'CO2': CO2,
                }

    features = pd.DataFrame(data, index=[0])




    model = load_model('models/drugs/drugRS_v1.h5')

    labels = ['drug__AMLODIPINE', 'drug__ANALGESICBALM', 'drug__ASPIRIN',
                  'drug__ATENOLOL', 'drug__ATORVASTATIN', 'drug__CALCIUMCARBONATE',
                  'drug__DOXAZOSIN', 'drug__ENALAPRIL', 'drug__FUROSEMIDE',
                  'drug__GLIPIZIDE', 'drug__HYDROCHLOROTHIAZIDE', 'drug__LORAZEPAM',
                  'drug__LOSARTAN', 'drug__METFORMIN',
                  'drug__METHYLSALICYLATE+MENTHOL+CAMPLOR', 'drug__OMEPRAZOLE',
                  'drug__PARACETAMOL', 'drug__PIOGLITAZONE', 'drug__SIMVASTATIN',
                  'drug__VITAMINBCOMPLEX']

    data = features.values.tolist()

    y_pred = model.predict(data)

    y_pred = list(y_pred[0]*100)

    result = pd.DataFrame({'Drug': labels, 'Accuracy': y_pred})
    result['Drug'] = result['Drug'].str.replace('drug__', '')
    result['Accuracy'] = result['Accuracy'].astype('float').round(2)
    result = result.sort_values(by=['Accuracy'], ascending=False)

    st.subheader('The recommended drugs are as belows:')

    ls_high = list(result['Drug'][result['Accuracy'] >= 60])
    ls_high = str(ls_high)
    st.write('> Highly recommended: ' + ls_high)

    ls_mid = list(result['Drug'][(result['Accuracy'] < 60) & (result['Accuracy'] >= 20)])
    ls_mid = str(ls_mid)
    st.write('> Recommended: ' + ls_mid)

    ls_low = list(result['Drug'][result['Accuracy'] < 20])
    ls_low = str(ls_low)
    st.write('> Rarely recommended: ' + ls_low)

    fig = plt.figure(figsize=(20, 10))
    plt.title('Drug vs Accuracy')
    sns.barplot(x='Drug', y='Accuracy', data=result, palette='hot')
    plt.ylabel('Accuracy')
    plt.xlabel('Drug')
    plt.xticks(rotation=65)

    st.subheader('The probability graph of prediction:')
    st.pyplot(fig)

