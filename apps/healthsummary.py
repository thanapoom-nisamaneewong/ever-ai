import streamlit as st
import streamlit.components.v1 as components

def app():

    #st.title('Health Summary')
    #st.write('Dashboard :')
    #st.components.v1.html("""<iframe width="1000" height="780" src="https://datastudio.google.com/embed/reporting/32b94f6a-c615-4a54-81ac-f59cfdcca1ec/page/GN3fC" frameborder="0" style="border:0" allowfullscreen></iframe>""",width=1400, height=800,scrolling=True)
    components.iframe("https://ever-ai-healthdashboard.herokuapp.com",width=1050, height=500, scrolling=True)
    st.write(' ')
    st.write("Patient Journeys :  check out this [link](http://neo4j.ai-nonprod.everapp.io:7474/browser/)")
    st.image("https://res.cloudinary.com/dsifak7nw/image/upload/v1639709766/kg_oiobxp.jpg")



