import streamlit as st
from streamlit.logger import get_logger
import streamlit.components.v1 as components
import datetime

LOGGER = get_logger(__name__)

thedate = datetime.date.today()
def run():

    st.write("""
    # Welcome to House Price Prediction!
    """) 

    st.write("###### Date: ", thedate)
   
                                          


if __name__ == "__main__":
    run()
