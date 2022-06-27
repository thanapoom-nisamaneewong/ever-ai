import streamlit as st
import streamlit.components.v1 as components

def app():
    st.components.v1.html("""<iframe width="1000" height="800" src="https://datastudio.google.com/embed/reporting/0aa1bdb6-0b5a-4aca-b17d-cd45a07a3782/page/cVJwC" frameborder="0" style="border:0" allowfullscreen></iframe>""",width=1400, height=800,scrolling=True)



