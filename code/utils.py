import os
import numpy as np
import argparse
import torch
import random

from dotenv import load_dotenv
from datasets import load_dataset

DEBUG_MODE = False

# seed 고정
def seed_fix(SEED=456):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def set_debug_mode(debug_mode):
    global DEBUG_MODE
    DEBUG_MODE = debug_mode
    
    debug_print("DEBUG_MODE가 설정되었습니다.")
    if not DEBUG_MODE:
        import warnings
        warnings.filterwarnings(action='ignore')
    

def debug_print(text):
    if DEBUG_MODE:
        print(text)

# config parser로 가져오기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml"
    )  # 입력 없을 시, 기본값으로 config.yaml을 가져옴
    return parser.parse_args()


# config 확인 (print)
def config_print(config, depth=0):
    if depth==0:
        print("*" * 40)
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if type(v) == dict:
            print(*prefix)
            config_print(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)
    if depth==0:
        print("*" * 40)


def wandb_name(train_file_name, train_lr, train_batch_size, test_size, user_name):
    data_name = train_file_name
    lr = train_lr
    bs = train_batch_size
    ts = test_size
    user_name = user_name
    return f"{user_name}_{data_name}_{lr}_{bs}_{ts}"


def load_env_file(filepath=".env"):
    try:
        # .env 파일 로드 시도
        if load_dotenv(filepath):
            debug_print(f".env 파일을 성공적으로 로드했습니다: {filepath}")
        else:
            raise FileNotFoundError  # 파일이 없으면 예외 발생
    except FileNotFoundError:
        debug_print(f"경고: 지정된 .env 파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        debug_print(f"오류 발생: .env 파일 로드 중 예외가 발생했습니다: {e}")


def check_dataset(hf_organization, hf_token, train_file_name):
    """
    로컬에 데이터셋 폴더가 없으면 Hugging Face에서 데이터를 다운로드하여 로컬에 CSV로 저장하는 함수.
    데이터셋을 로컬에 저장만 하고 반환값은 없습니다.

    Parameters:
    - hf_organization (str): Hugging Face Organization 이름
    - hf_token (str): Hugging Face 토큰
    - train_file_name (str): 로컬에 저장할 train file 이름
    - dataset_repo_id (str): Hugging Face에 저장된 데이터셋 리포지토리 ID (기본값: "datacentric-orginal")
    """
    # Define the folder path and file paths
    folder_path = os.path.join("..", "data")
    train_path = os.path.join(folder_path, "train.csv")

    # Check if local data folder exists
    if not os.path.exists(train_path):
        debug_print(
            f"로컬에 '{train_path}' 데이터가 존재하지 않습니다.허깅페이스에서 다운로드를 시도합니다."
        )

        # Load dataset from Hugging Face if local folder is missing
        full_repo_id = f"{hf_organization}/datacentric-{train_file_name}"
        dataset = load_dataset(full_repo_id, split="train", token=hf_token)

        # 데이터셋을 CSV로 저장
        dataset.to_pandas().to_csv(train_path, index=False)
        debug_print(f"데이터셋이 '{train_path}'에 다운로드되었습니다.")
    else:
        debug_print(f"로컬파일을 로드합니다.")