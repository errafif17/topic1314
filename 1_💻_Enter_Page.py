import streamlit as st
from streamlit.logger import get_logger
import streamlit.components.v1 as components

LOGGER = get_logger(__name__)

def run():
    st.image(r'./enter_page_image.jpg', use_column_width=True)
    # st.set_page_config(page_title="Enter Page", page_icon="💻")

    st.write("""
    # Welcome to House Price Prediction!
    """) 

    st.markdown(
        """
    This repo has been developed for the Istanbul Data Science Bootcamp, organized in cooperation with IBB & Kodluyoruz. 
    Prediction for house prices was developed using the Kaggle House Prices - Advanced Regression Techniques competition dataset.
    
    ### Goal

    The goal of this project is to predict the price of a house in Ames using the features provided by the dataset.
    
    ------

    ###### Group 2 | Machine Learning Model Deployment

    ###### Version: 1.0
    """
    )
                                    
                                          
                            

if __name__ == "__main__":
    run()
