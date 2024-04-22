import streamlit as st
from dotenv import dotenv_values
import google.generativeai as genai

class CommentAnalyser:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_summary(self, comments):
        try:
            response = self.model.generate_content(f'Please summarise this document: {comments}')
            return response.text
        except Exception as e:
            st.error(f"Failed to generate summary: {str(e)}")
            return None

    def analyze_viewpoints(self, comments):
        try:
            response = self.model.generate_content(f'Perform Vectorisation and group similar viewpoints on ```{comments}```')
            return response.text
        except Exception as e:
            st.error(f"Failed to analyze viewpoints: {str(e)}")
            return None

    def analyze_sentiments(self, comments):
        try:
            response = self.model.generate_content(f'Perform sentiment analysis on ```{comments}```')
            return response.text
        except Exception as e:
            st.error(f"Failed to analyze sentiments: {str(e)}")
            return None