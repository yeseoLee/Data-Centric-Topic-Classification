import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def restore_headline(noisy_headline):
    prompt = f"""다음은 노이즈가 있는 한국어 뉴스 기사 제목입니다. 이를 원래의 정확한 한국어 제목으로 복원해주세요.
    단계별로 복원 과정을 설명하고, 최종 복원된 제목을 제시해주세요.

    노이즈 있는 제목: {noisy_headline}

    복원 과정:
    1. 노이즈 단어 제거:
    2. 문법 및 맞춤법 검사:

    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    restored_process = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return restored_process.split("복원 과정:")[-1].strip()

def batch_restore_headlines(noisy_headlines):
    results = []
    for headline in noisy_headlines:
        restored = restore_headline(headline)
        results.append((headline, restored))
    return results

# 테스트
noisy_headlines = [
    "pI美(미국)대선I앞두고 R2fr단 발 비해 감시 강화",
    "정i 파1 미사z KT 이용기간 2e 단 Q분종U",
    "아이XSI수리0 b대 맥3 디dF레 41"
]

restored_results = batch_restore_headlines(noisy_headlines)

for original, restored in restored_results:
    print(f"노이즈 있는 제목: {original}")
    print(f"복원 과정:\n{restored}")
    print("-" * 50)
