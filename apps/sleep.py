import streamlit as st
import streamlit.components.v1 as components
def app():
    st.title('Sleep Lab')
    st.components.v1.html("""<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQwjzbwo-1S0BLcGd-n8v55T_QHCl0eUqbFqPITJsC9E1pmkFQVc5dC0FFPTnX6Tw/embed?start=false&loop=false&delayms=3000" frameborder="0" width="640" height="389" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>""",width=700, height=500,scrolling=True)
    st.components.v1.html("""<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vR6LBtRa2NevTfdQD9pQkwsap5iMGjXJcZdkBpH3inC2dOn-DnDSfwVujSUI_er-w/embed?start=false&loop=false&delayms=3000" frameborder="0" width="640" height="389" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>""",width=700, height=500,scrolling=True)
