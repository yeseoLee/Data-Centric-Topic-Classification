import csv
import random
import re
from collections import defaultdict

import torch
from prompt import get_prompt_synonyms
from tqdm import tqdm  # Import tqdm for progress bar
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
PROMPT_VERSION = 1

# 모델과 토크나이저 로드
model_name = MODEL_NAME
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 사용 가능 시 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def get_synonyms(word, model, tokenizer, device, max_length=100, num_return_sequences=1):
    prompt = get_prompt_synonyms(PROMPT_VERSION, word)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=num_return_sequences)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    synonyms = re.findall(r":\s*(.*?)(?:\n|$)", response)
    if synonyms:
        return [syn.strip() for syn in synonyms[0].split(",") if syn.strip()]
    return []


def augment_text(text, model, tokenizer, device, random_ratio=0.3):
    words = text.split()
    augmented_words = []
    for word in words:
        if random.random() < random_ratio:  # 30% 확률로 단어를 동의어로 대체
            synonyms = get_synonyms(word, model, tokenizer, device)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


def main(
    input_file="../data/aug_syn_input.csv",
    output_file="../data/aug_syn_output.csv",
    min_row=3,
    samples_per_class=10,
):
    """
    데이터 증강을 수행하는 메인 함수

    Args:
        input_file (str): 입력 CSV 파일의 경로.
        output_file (str): 증강된 데이터를 저장할 CSV 파일의 경로.
        min_row (int): 데이터로 사용할 최소 행의 개수. 이보다 적은 행을 가진 데이터는 건너뜀.
        samples_per_class (int): 각 타겟 클래스당 증강할 샘플의 수.

    Returns:
        None: 결과를 output_file에 저장
    """

    # 타겟 클래스별 데이터 저장을 위한 딕셔너리 초기화
    data_by_target = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 헤더 읽기

        for row in reader:
            if len(row) < min_row:  # 데이터가 부족한 경우 건너뛰기
                continue
            id, text, target = row
            data_by_target[target].append((id, text))

    # CSV 파일에 증강된 데이터 쓰기
    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # 헤더 쓰기

        for target, samples in data_by_target.items():
            # 각 타겟 클래스에서 샘플을 무작위로 선택하고 증강 수행
            selected_samples = random.sample(samples, min(samples_per_class, len(samples)))

            for id, text in tqdm(selected_samples, desc=f"Processing Target {target}"):
                writer.writerow([id, text, target])  # 원본 데이터 쓰기

                # 증강된 데이터 생성 및 쓰기
                augmented_text = augment_text(text, model, tokenizer, device)
                writer.writerow([f"{id}_aug", augmented_text, target])

    print(f"데이터 증강이 완료되었습니다. 결과가 '{output_file}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    main()
