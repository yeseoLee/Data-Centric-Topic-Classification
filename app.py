import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 요번 프로젝트에서 뭐가 중요한지를 계속 생각하면서 무엇을 어떻게 띄울것인지를 생각하자.

load_file_path = "./data/train.csv" # 시각화 할 데이터 파일 경로

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(load_file_path)

df = load_data()

st.title("데이터 시각화 대시보드")

# Filter data by target value
st.header("라벨값 기준으로 데이터 살펴보기")
target_value = st.selectbox("label :", df['target'].unique())
filtered_data = df[df['target'] == target_value]
st.write(filtered_data)

# Bar plot
st.header("라벨값 기준 데이터 수 Bar plot")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
ax.set_xlabel('Label')
ax.set_ylabel('# of data')
st.pyplot(fig)
for i in range(6):
    st.write(f"label {i} : {df[df['target'] == i].shape[0]}개")
st.write(f"전체 데이터 수 : {df.shape[0]}개")





