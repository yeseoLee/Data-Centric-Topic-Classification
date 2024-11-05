# Load model directly
# pip install git+https://github.com/huggingface/transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("beomi/KoAlpaca-llama-1-7b")
model = LlamaForCausalLM.from_pretrained("beomi/KoAlpaca-llama-1-7b")


def restore_headline(noisy_headline):
    # 프롬프트 생성
    prompt = f"""다음은 노이즈가 있는 한국어 뉴스 기사 제목입니다. 이를 원래의 정확한 한국어 제목으로 복원해주세요:

    노이즈 있는 제목: {noisy_headline}

    복원된 제목:"""

    # 입력 인코딩
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # 모델 출력 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 생성할 최대 토큰 수
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    # 출력 디코딩 및 처리
    restored_headline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 프롬프트 제거 및 결과 정제
    restored_headline = restored_headline.split("복원된 제목:")[-1].strip()

    return restored_headline


# 테스트
noisy_headlines = ["프로야구롯TKIAs광주 경기 y천취소"]

for noisy_headline in noisy_headlines:
    restored = restore_headline(noisy_headline)
    print(f"노이즈 있는 제목: {noisy_headline}")
    print(f"복원된 제목: {restored}")
    print()
