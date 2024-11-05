import sentencepiece as spm
import pandas as pd
import csv

# 1. SentencePiece 모델 학습
def train_spm_model(csv_file, model_prefix='korean_news', vocab_size=8000):
    # CSV 파일에서 'text' 열만 추출하여 임시 파일로 저장
    df = pd.read_csv(csv_file)
    with open('temp_text.txt', 'w', encoding='utf-8') as f:
        for text in df['text']:
            f.write(f"{text}\n")
    
    # SentencePiece 모델 학습
    spm.SentencePieceTrainer.train(
        input='temp_text.txt',
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='unigram',
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<pad>', '<unk>', '<bos>', '<eos>']
    )

# 2. 텍스트 토큰화 함수
def tokenize_text(text, sp):
    return ' '.join(sp.encode_as_pieces(text))

# 3. CSV 파일 처리 및 토큰화
def process_csv(input_file, output_file, sp):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['tokenized_text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row['tokenized_text'] = tokenize_text(row['text'], sp)
            writer.writerow(row)

# 메인 실행 코드
if __name__ == "__main__":
    input_file = 'your_input_file.csv'  # 입력 CSV 파일 이름
    output_file = 'tokenized_output.csv'  # 출력 CSV 파일 이름
    model_prefix = 'korean_news'

    # 1. SentencePiece 모델 학습
    train_spm_model(input_file, model_prefix)

    # 2. 학습된 모델 로드
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')

    # 3. CSV 파일 처리 및 토큰화
    process_csv(input_file, output_file, sp)

    print(f"토큰화된 결과가 {output_file}에 저장되었습니다.")