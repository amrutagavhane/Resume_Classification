# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 23:23:54 2023

@author: admin
"""

import streamlit as st

def main():
    st.set_page_config(page_title="Resumes Classification", page_icon="📑")

    st.title("Resumes Classification")
    st.markdown("Here all the resumes in the form of PDF, DOCX, DOC, TXT.")

    # Add your Streamlit app's content here, if any.

    # Resumes button
    st.markdown("<h3>Resumes</h3>", unsafe_allow_html=True)
    st.button("Resumes")

    # Code button
    st.markdown("<h3>Code</h3>", unsafe_allow_html=True)
    st.button("Code")

if __name__ == "__main__":
    main()
