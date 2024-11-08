"""
denoise.py 를 실행후 processing.py로 후처리 해주세요
"""

import csv
import logging
import re

import pandas as pd
import torch
from prompt import get_prompt_denoise
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
PROMPT_VERSION = 1


pd.set_option("display.max_rows", None)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 모델 및 토크나이저 로드
logging.info("모델 및 토크나이저 로딩 중...")
model_name = MODEL_NAME
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 사용 가능 시 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logging.info(f"모델을 {device}로 이동했습니다.")


def restore_headline(noisy_headline):
    prompt = get_prompt_denoise(PROMPT_VERSION, noisy_headline)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    restored_headline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "복원된 제목:" 이후의 텍스트를 추출
    if "복원된 제목:" in restored_headline:
        restored_headline = restored_headline.split("복원된 제목:")[-1].strip()

    # "답변:" 부분 제거
    restored_headline = re.sub(r"^답변:\s*", "", restored_headline)

    # 필요에 따라 추가적인 처리...

    return restored_headline.strip()


# CSV 파일 읽기, 데이터 타입 지정
input_file = "../data/21_relabeling_2800.csv"
logging.info(f"CSV 파일 '{input_file}' 읽는 중...")
df = pd.read_csv(
    input_file,
    quoting=csv.QUOTE_ALL,
    dtype={"ID": str, "text": str, "target": int, "is_noise": int},
)
logging.info(f"총 {len(df)} 개의 행을 읽었습니다.")

# is_noise가 1인 항목 필터링
noisy_df = df[df["is_noise"] == 1]
logging.info(f"노이즈가 있는 {len(noisy_df)} 개의 행을 찾았습니다.")

# 초반 100개 항목만 선택
noisy_df_subset = noisy_df.head(1602)

# 디노이징 수행
logging.info("디노이징 작업 시작...")
tqdm.pandas()
noisy_df_subset["denoised_text"] = noisy_df_subset["text"].progress_apply(restore_headline)
logging.info("디노이징 작업 완료.")

# 결과 DataFrame 생성 (is_noise 열은 포함하지 않음)
result_df = noisy_df_subset[["ID", "denoised_text", "target"]]
result_df = result_df.rename(columns={"denoised_text": "text"})

output_file = "../data/semi-final3.csv"
logging.info(f"결과를 '{output_file}'에 저장 중...")
result_df.to_csv(output_file, index=False, encoding="utf-8")

# is_noise가 0인 항목 필터링
non_noisy_df = df[df["is_noise"] == 0]
logging.info(f"is_noise가 0인 {len(non_noisy_df)} 개의 행을 찾았습니다.")

# 두 DataFrame 합치기
final_df = pd.concat([result_df, non_noisy_df[["ID", "text", "target"]]], ignore_index=True)

# 데이터프레임 섞기
final_df = final_df.sample(frac=1).reset_index(drop=True)

# 결과를 CSV 파일로 저장 (is_noise 열은 포함하지 않음)
output_file = "../data/final3.csv"
logging.info(f"결과를 '{output_file}'에 저장 중...")
final_df.to_csv(output_file, index=False, encoding="utf-8")

logging.info(f"디노이징 완료. 결과가 '{output_file}' 파일에 저장되었습니다.")
