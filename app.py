# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:47:27 2023

@author: admin
"""

import streamlit as st

# Set page title
st.set_page_config(page_title="Deployment", page_icon=":rocket:")

# Define the main function
def main():
    # Set background image
    st.markdown(
        """
        <style>
        body {
            background-image: url('2.png');
            background-repeat: no-repeat;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main header
    st.markdown("<h1 style='text-align: center; color: #170317; margin-top: 200px;'>Resumes Classification</h1>", unsafe_allow_html=True)

    # Information section
    st.markdown(
        """
        <div style='background-color: #E6A7D3; width: 300px; border: 15px solid green; padding: 50px; margin: auto; margin-top: 10px;'>
        Here all the resumes in the form of PDF, DOCX, DOC, TXT.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Resumes button
    st.markdown("<a href='Resumes/'><button style='background-color: #8dbbd7; border: none; color: #1F1C1C; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; margin-left: 480px;'>Resumes</button></a>", unsafe_allow_html=True)

    # Code button
    st.markdown("<a href='Resume_project123.ipynb'><button style='background-color: #8dbbd7; margin-left: 250px;'>Code</button></a>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
