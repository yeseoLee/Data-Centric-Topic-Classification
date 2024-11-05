"""
1. CSV 파일에서 노이즈가 있는 데이터와 없는 데이터를 분리
2. 노이즈가 없는 데이터를 사용하여 SentencePiece 모델 훈련
3. 훈련된 SentencePiece 모델을 사용하여 노이즈가 있는 데이터 토큰화
4. SentencePiece 모델을 사용하여 토큰화된 데이터 디노이징

출력:
- noise_data.csv: 노이즈가 있는 데이터
- non_noise_data.csv: 노이즈가 없는 데이터
- sentencepiece_model.model: 훈련된 SentencePiece 모델
- denoised_data.csv: 디노이징된 데이터

주의: 이 스크립트는 입력 CSV 파일에 'text'와 'is_noise' 열이 있다고 가정합니다.
"""

import os

import pandas as pd
import sentencepiece as spm


# 데이터 파일 경로
data_file = "/content/2_base_2800_noise_detected.csv"


def split_noise_data(data_file):
    # CSV 파일 읽기
    df = pd.read_csv(data_file)

    # 노이즈가 있는 데이터와 없는 데이터 분리
    noise_data = df[df["is_noise"] == 1]
    non_noise_data = df[df["is_noise"] == 0]

    # 분리된 데이터를 CSV 파일로 저장
    noise_data.to_csv("noise_data.csv", index=False)
    non_noise_data.to_csv("non_noise_data.csv", index=False)

    print("데이터가 성공적으로 분리되어 CSV 파일로 저장되었습니다.")
    return "non_noise_data.csv", "noise_data.csv"


def train_sentencepiece_model(df, model_prefix, vocab_size=19931):
    # 모든 텍스트 데이터로 임시 파일 생성
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        for text in df["text"]:
            f.write(text + "\n")

    # SentencePiece 모델 훈련
    spm.SentencePieceTrainer.train(
        input="temp_corpus.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
    )

    # 임시 파일 제거
    os.remove("temp_corpus.txt")


def load_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def tokenize_and_detokenize(text, sp):
    # 텍스트를 토큰화하고 다시 디토큰화하여 반환
    tokens = sp.encode_as_pieces(text)
    return sp.decode_pieces(tokens)


def process_dataframe(df, sp):
    df["denoised_text"] = df["text"].apply(lambda x: tokenize_and_detokenize(x, sp))
    return df


# 메인 실행 코드
if __name__ == "__main__":
    # 노이즈 데이터 분리
    non_noise_file, noise_file = split_noise_data(data_file)

    # SentencePiece 모델 훈련을 위한 설정
    model_prefix = "sentencepiece_model"

    # 노이즈가 없는 데이터 읽기
    df = pd.read_csv(non_noise_file)

    # SentencePiece 모델 훈련
    train_sentencepiece_model(df, model_prefix)

    # 훈련된 SentencePiece 모델 로드
    sp = load_sentencepiece_model(f"{model_prefix}.model")

    # 노이즈가 있는 새로운 데이터 읽기
    noise_df = pd.read_csv(noise_file)

    # 데이터 처리 및 디노이징
    denoised_df = process_dataframe(noise_df, sp)

    # 결과를 CSV 파일로 저장
    denoised_df.to_csv("denoised_data.csv", index=False)

    print("모든 처리가 완료되었습니다.")
