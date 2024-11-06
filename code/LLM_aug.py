import re
import csv
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 사용 가능 시 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_article(title):
    prompt = f"""다음 기사 제목에 대한 내용을 작성해주세요.
    주제는 똑같지만 기사 내용은 창의적이어도 좋습니다.
    : {title}

    기사 내용:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("기사 내용:")[1].strip()

def generate_new_title(article):
    prompt = f"""다음 기사 내용을 바탕으로 새로운 제목을 생성해주세요.
    절대 기사 내용에 들어간 단어는 쓰지 마세요.:
    \n\n{article}\n\n새로운 제목:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=20, num_return_sequences=1, no_repeat_ngram_size=2, temperature=1.5
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("새로운 제목:")[-1].strip()

def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "text", "target"])
        for i, (text, target) in enumerate(data):
            cleaned_text = re.sub(r'^"|"$', '', text)
            cleaned_text = re.sub(r'\s*이\s*제목은.*$', '', cleaned_text)
            cleaned_text = ' '.join(cleaned_text.split())
            writer.writerow([f"ynat-v1_train_{i:05d}", cleaned_text, target])

def process_title(title, target):
    article = generate_article(title)
    new_title = generate_new_title(article)
    return article, new_title, target

def read_csv(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        return [(row[1], row[2]) for row in reader]  # (text, target) 튜플 리스트 반환

def process_csv(input_filename, output_filename, max_rows=None):
    data = read_csv(input_filename)
    
    # 텍스트 길이에 따라 정렬
    data.sort(key=lambda x: len(x[0]), reverse=True)
    
    if max_rows:
        data = data[:max_rows]

    processed_data = []
    for i, (title, target) in enumerate(data):
        article, new_title, _ = process_title(title, target)

        print(f"\n원래 제목: {title}")
        print(f"생성된 기사 내용:\n{article}")
        print(f"새로운 제목: {new_title}")
        print("-" * 50)

        processed_data.append((new_title, target))

    save_to_csv(processed_data, output_filename)
    print(f"처리 완료. 결과가 '{output_filename}' 파일에 저장되었습니다.")

# 사용 예시
input_csv = "../data/semi-final4.csv"  # 입력 CSV 파일명
output_csv = "generated_titles4.csv"  # 출력 CSV 파일명
max_rows = 600  # 처리할 최대 행 수 (None으로 설정하면 모든 행 처리)

process_csv(input_csv, output_csv, max_rows)