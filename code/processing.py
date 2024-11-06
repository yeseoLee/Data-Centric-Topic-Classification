import pandas as pd


# CSV 파일 읽기
df = pd.read_csv("../data/final4.csv")


# text 열에서 \n 이후의 내용을 제거하는 함수 정의
def clean_text(text):
    if isinstance(text, str):  # text가 문자열인 경우에만 처리
        return text.split("\n")[0]  # \n 이전의 텍스트만 반환
    return text  # 문자열이 아닌 경우 원래 값을 반환


# clean_text 함수를 text 열에 적용
df["text"] = df["text"].apply(clean_text)

# 결과를 새로운 CSV 파일로 저장
df.to_csv("../data/final5.csv", index=False, encoding="utf-8")
