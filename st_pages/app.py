import cleanlab_noize_viz
import data_overview
import noise_viz
import streamlit as st
from data_loader import load_data

# 요번 프로젝트에서 뭐가 중요한지를 계속 생각하면서 무엇을 어떻게 띄울것인지를 생각하자.

st.title("데이터 시각화 대시보드")
df = load_data()

page = st.sidebar.selectbox(
    "페이지 선택",
    ["단순 데이터 시각화", "노이즈 비율 시각화", "클린랩 노이즈 비율 시각화"],
)

if page == "단순 데이터 시각화":
    data_overview.show(df)

elif page == "노이즈 비율 시각화":
    noise_viz.show(df)

elif page == "클린랩 노이즈 비율 시각화":
    uploaded_file = st.file_uploader(
        "CSV 파일을 드래그 앤 드롭하거나 선택하세요", type="csv"
    )

    if uploaded_file is not None:
        cleanlab_noize_viz.show(uploaded_file)
        cleanlab_noize_viz.visualization(uploaded_file)
    else:
        st.write("파일을 업로드해주세요.")
