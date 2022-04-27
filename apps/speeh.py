import streamlit as st
from bokeh.models.widgets import Button

from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import time
def app():

    st.markdown(""" <style> .font1 {
    font-size:40px ; font-family: "Gill Sans", sans-serif; color: darkgoldenrod;}
    </style> """, unsafe_allow_html=True)

    st.markdown('<p class="font1"><b>Text to Voice</b></p>', unsafe_allow_html=True)
    lang1 = st.radio(
         "Please select a language. ",
         ('English', 'Thai'))
    text = st.text_input("Say what ?")


    tts_button = Button(label="Listen", width=100,button_type="success")

    if lang1 =='Thai':


        tts_button.js_on_event("button_click", CustomJS(code=f"""
        var u = new SpeechSynthesisUtterance();
        u.text = "{text}";
        u.lang = 'th';
    
        speechSynthesis.speak(u);
        """))
        st.bokeh_chart(tts_button)

    if lang1 =='English':
        tts_button.js_on_event("button_click", CustomJS(code=f"""
        var u = new SpeechSynthesisUtterance();
        u.text = "{text}";
        u.lang = 'en-US	';
    
        speechSynthesis.speak(u);
        """))
        st.bokeh_chart(tts_button)



    st.markdown(""" <style> .font2 {
    font-size:40px ; font-family: "Gill Sans", sans-serif; color: darkblue; }
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2"><b>Voice to Text</b></p>', unsafe_allow_html=True)
    #
    lang = st.radio(
         "Please select a language.",
         ('English', 'Thai'))
    stt_button = Button(label="Speak", width=100,button_type="success")
    if lang =='English':
        stt_button.js_on_event("button_click",CustomJS(code="""
        alert("Click 'ok' to start recording your voice.")"""))

        stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US'
        
        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                recognition.stop()
            }
        }
        recognition.start();
        """))


        result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=False,
            override_height=75,
            debounce_time=0)

        if result:
            if "GET_TEXT" in result:
                st.markdown(f"<font color='black' size=4 ><b>result : </b></font><font color='blue' size=4 ><b>{result.get('GET_TEXT')}</b></font>",unsafe_allow_html=True)







    if lang =='Thai':
        stt_button.js_on_event("button_click",CustomJS(code="""
        alert("Click 'ok' to start recording your voice.")"""))


        stt_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang ='th-TH';
        
    
        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if ( value != "") {
                document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                recognition.stop()
            }
        }
        recognition.start();
        """))


        result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=True,
            override_height=75,
            debounce_time=0)


        if result:

            if "GET_TEXT" in result:
                st.markdown(f"<font color='black' size=4 ><b>result : </b></font><font color='blue' size=4 ><b>{result.get('GET_TEXT')}</b></font>",unsafe_allow_html=True)

