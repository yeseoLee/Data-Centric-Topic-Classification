import pandas as pd

# 파일 경로
test_with_predictions_path = "../../data/test_with_predictions.csv"
train_path = "../../data/train.csv"
output_path = "../../data/train_cleaning_AE.csv"

# 파일 불러오기
test_with_predictions = pd.read_csv(test_with_predictions_path)
train = pd.read_csv(train_path)

# ID를 기준으로 test_with_predictions의 target으로 덮어씌우기
# train 데이터프레임에서 test_with_predictions에 있는 ID들의 target 값을 덮어씌움
train.loc[train['ID'].isin(test_with_predictions['ID']), 'target'] = \
    train.loc[train['ID'].isin(test_with_predictions['ID']), 'ID'].map(
        test_with_predictions.set_index('ID')['target']
    )

# 덮어씌운 train 데이터프레임을 새로운 파일로 저장
train.to_csv(output_path, index=False)

print("train_cleaning 파일이 생성되었습니다.")