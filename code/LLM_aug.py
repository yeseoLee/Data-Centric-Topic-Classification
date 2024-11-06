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
    기사 내용에 들어간 단어의 동의어로 대체해서 생성하세요.:
    \n\n{article}\n\n새로운 제목:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=20, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.9
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("새로운 제목:")[-1].strip()


def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "text", "target"])
        for i, (text, target) in enumerate(data):
            # 불필요한 쌍따옴표와 설명 텍스트 제거
            cleaned_text = re.sub(r'^"|"$', '', text)
            cleaned_text = re.sub(r'\s*이\s*제목은.*$', '', cleaned_text)
            # 줄바꿈 문자를 공백으로 대체하고 앞뒤 공백 제거
            cleaned_text = ' '.join(cleaned_text.split())
            writer.writerow([f"ynat-v1_train_{i:05d}", cleaned_text, target])


def process_title(title, target):
    article = generate_article(title)
    new_title = generate_new_title(article)
    return article, new_title, target


# 여러 개의 초기 제목 준비
initial_titles = [
    "추신수 타율 0.265로 시즌 마감 최지만은 19홈런·6",
    "KBO 리그, 외국인 선수 규정 변경 검토 중",
    "공사업체 협박에 분쟁해결 명목 돈 받은 언론인 집행유예",
    "서울에 다시 오존주의보 도심·서북·동북권 발령",
    "크루즈 관광객용 반나절 부산 해안 트레킹 상품 개발",
]
target = 4  # 예시 target 값

processed_data = []
for i in range(5):  # 5개의 예시 생성
    title = random.choice(initial_titles)
    article, new_title, target = process_title(title, target)

    print(f"\n원래 제목: {title}")
    print(f"생성된 기사 내용:\n{article}")
    print(f"새로운 제목: {new_title}")
    print("-" * 50)

    processed_data.append((new_title, target))

save_to_csv(processed_data, "generated_titles.csv")
print("처리 완료. 결과가 'generated_titles.csv' 파일에 저장되었습니다.")
