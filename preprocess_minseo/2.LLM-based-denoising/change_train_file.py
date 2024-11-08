import pandas as pd


# 파일 경로
model_result = "../denoised_results_relabel.csv"
train_path = "../../data/train_cleaning_label_5sj.csv"
output_path = "../../data/train_cleaning_label_5sj_denoise_add_relabel.csv"

# 파일 불러오기
test_with_predictions = pd.read_csv(model_result)
train = pd.read_csv(train_path)


# # 교체하기
# # test_with_predictions의 'text'로 train 데이터의 'text' 열 덮어씌우기
# train.loc[train["ID"].isin(test_with_predictions["ID"]), "text"] = train["ID"].map(
#     test_with_predictions.set_index("ID")["denoised_text"]
# )

# 추가하기
# # test_with_predictions의 기존 text 열 삭제
# test_with_predictions = test_with_predictions.drop(columns=["text"])
# # 'denoised_text' 열 이름을 'text'로 변경
# test_with_predictions = test_with_predictions.rename(columns={"denoised_text": "text"})
# train 데이터프레임과 test_with_predictions 데이터프레임을 단순히 아래로 쌓기
train_combined = pd.concat([train, test_with_predictions], ignore_index=True)

# 덮어씌운 train 데이터프레임을 새로운 파일로 저장
train_combined.to_csv(output_path, index=False)

print("train_cleaning 파일이 생성되었습니다.")
