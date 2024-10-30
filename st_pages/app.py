import streamlit as st
from data_loader import load_data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

# 요번 프로젝트에서 뭐가 중요한지를 계속 생각하면서 무엇을 어떻게 띄울것인지를 생각하자.

st.title("데이터 시각화 대시보드")
df = load_data()

page = st.sidebar.selectbox(
    "페이지 선택",
    [
        "단순 데이터 시각화",
        "노이즈 비율 시각화",
    ],
)

if page == "단순 데이터 시각화":
    import data_overview

    data_overview.show(df)

elif page == "노이즈 비율 시각화":
    import noise_viz

    noise_viz.show(df)
