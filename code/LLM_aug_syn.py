import re
import csv
import random
import torch
from tqdm import tqdm  # Import tqdm for progress bar
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# 모델과 토크나이저 로드
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 사용 가능 시 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_synonyms(word, model, tokenizer, device):
    prompt = f"다음 단어의 동의어를 쉼표로 구분하여 5개만 나열해주세요: {word}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    synonyms = re.findall(r':\s*(.*?)(?:\n|$)', response)
    if synonyms:
        return [syn.strip() for syn in synonyms[0].split(',') if syn.strip()]
    return []

def augment_text(text, model, tokenizer, device):
    words = text.split()
    augmented_words = []
    for word in words:
        if random.random() < 0.3:  # 30% 확률로 단어를 동의어로 대체
            synonyms = get_synonyms(word, model, tokenizer, device)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# CSV 파일 읽기 및 증강
input_file = '../data/semi-final4.csv'
output_file = '../data/augmented_output.csv'
max_rows_to_process = 1  # 처리할 최대 행 수

# 각 타겟 클래스에 대해 증강할 샘플 수 설정
samples_per_class = 10

# 타겟 클래스별 데이터 저장을 위한 딕셔너리 초기화
data_by_target = defaultdict(list)

with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # 헤더 읽기
    
    for row in reader:
        if len(row) < 3:  # 데이터가 부족한 경우 건너뛰기
            continue
        id, text, target = row
        data_by_target[target].append((id, text))

# CSV 파일에 증강된 데이터 쓰기
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
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

print("데이터 증강이 완료되었습니다. 결과가 'augmented_output.csv' 파일에 저장되었습니다.")