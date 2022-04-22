import streamlit as st

def app():
    st.title('Diagnosis Prediction based on Clinical Text')
    st.markdown(' ')
    st.markdown("<font color='black' size=5 ><b>Chief complaint:</b></font>",unsafe_allow_html=True)
    st.write("check out this [link](http://54.184.187.240:8501/)")
    st.markdown(' ')
    st.markdown("<font color='black' size=5 ><b>Chief complaint with patient profile :</b></font>",unsafe_allow_html=True)
    st.write("check out this [link](https://ever-diagnosis.herokuapp.com/)")
