# 필요한 라이브러리 임포트
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
train_df = pd.read_csv("../data/df_morph_condition_V1_train.csv")
test_df = pd.read_csv("../data/df_morph_condition_V1_test.csv")

# 학습 데이터셋 준비 (train, validation으로 분할)
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)
train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)

# test 데이터셋 준비
test_data = Dataset.from_pandas(test_df)

# 데이터셋을 DatasetDict 형식으로 변환
dataset = DatasetDict({
    "train": train_data,
    "validation": val_data,
    "test": test_data
})

# 모델과 토크나이저 설정
model_name = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_df['target'].unique())).to(device)

# 데이터 전처리 함수
# 데이터 전처리 함수 수정
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True)
    if "target" in examples:  # train 데이터의 경우
        tokenized_inputs["labels"] = examples["target"]  # 'labels'에 타깃 라벨을 추가
    return tokenized_inputs


# 데이터셋에 전처리 적용
encoded_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 모델 학습
trainer.train()

# test 데이터에 대해 예측
predictions = trainer.predict(encoded_dataset["test"])
pred_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()

# test 데이터프레임에 예측 결과 추가
test_df["target"] = pred_labels

# 예측 결과를 포함한 test.csv 파일 저장
test_df.to_csv("../data/test_with_predictions.csv", index=False)

print("test_with_predictions.csv 파일이 생성되었습니다.")
