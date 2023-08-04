# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:11:46 2023

@author: admin
"""

import streamlit as st

def main():
    # Set page title and icon
    st.set_page_config(page_title="Deployment", page_icon="🚀")

    # CSS style for the background image
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
        unsafe_allow_html=True
    )

    # Title
    st.title("Resumes Classification")
    st.markdown("Here all the resumes in the form of PDF, DOCX, DOC, TXT.")

    # Div with green border
    st.markdown(
        """
        <div style="background-color: rgb(230, 167, 211); width: 300px;
            border: 15px solid green; padding: 50px; margin: auto;
            margin-top: 10px;">
            Here goes the content you want to display within the div.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Resumes button
    st.markdown("<h3>Resumes</h3>", unsafe_allow_html=True)
    st.button("Resumes")

    # Code button
    st.markdown("<h3>Code</h3>", unsafe_allow_html=True)
    st.button("Code")

if __name__ == "__main__":
    main()
