import streamlit as st
import streamlit.components.v1 as components
from multiapp import MultiApp
from apps import diabetes, hypertension, hyperlipidaemia,healthpredictionapp,diseasepredictionapp,drugapp,home,outbreakforcastingapp,pakinsondrawingsapp,druginterctionapp,ocr,xray,sleep,healthsummary,patho,covid,ekyc,map,speeh

app = MultiApp()
st.set_page_config(layout="wide")
st.markdown(""" <style> .font {
font-size:55px ; font-family: 'Arial'; color: #ffffff; background-image: linear-gradient(darkslategray, teal); text-align:center} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font"><b>EVER AI Products</b></p>', unsafe_allow_html=True)

# Add all your application here
app.add_app('Health Summary', healthsummary.app)
app.add_app('Health Prediction', healthpredictionapp.app)
app.add_app('Diagnosis Prediction based on Clinical Text', diseasepredictionapp.app)
app.add_app('Drug Recommendation System', drugapp.app)
app.add_app('Drug Interaction', druginterctionapp.app)
app.add_app('Outbreak Forecasting', outbreakforcastingapp.app)
app.add_app('Outbreak Map',map.app)
app.add_app('Parkinson Drawing Prediction', pakinsondrawingsapp.app)
app.add_app('OCR (Image / Handwriting to Text)', ocr.app)
app.add_app('Speech Recognition', speeh.app)
app.add_app('Chest X-ray Classification', xray.app)
app.add_app('Sleep Lab Classification', sleep.app)
app.add_app('Pathology Slide Classification', patho.app)
app.add_app('Covid-19 Outbreak Prediction in Thailand', covid.app)
app.add_app('eKYC - Facial Recognition', ekyc.app)
app.add_app(' ', home.app)
# The main app
app.run()
