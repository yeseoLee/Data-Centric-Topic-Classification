import pandas as pd
import streamlit as st


load_file_path = "../data/train.csv"  # 시각화 할 데이터 파일 경로


# Load data
@st.cache_data
def load_data():
    return pd.read_csv(load_file_path)
