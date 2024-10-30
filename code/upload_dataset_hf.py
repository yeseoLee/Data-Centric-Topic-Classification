import os
from huggingface_hub import HfApi
from datasets import load_dataset
from dotenv import load_dotenv
from main import load_env_file


def upload_dataset_folder_to_hub(foler_name, token, private=True):
    """
    폴더 내 데이터 파일들을 Hugging Face Hub에 데이터셋으로 업로드하는 함수.

    Parameters:
    - repo_id (str): Hugging Face의 'Organization/Repository' 형식의 데이터셋 이름
    - token (str): Hugging Face 액세스 토큰. 쓰기 권한 필요
    - folder_name (str): 업로드할 로컬 데이터 폴더 이름
    - private (bool): True면 비공개, False면 공개 설정

    Returns:
    - None
    """
    api = HfApi()
    repo_id = f"paper-company/datacentric-{foler_name}"

    # 리포지토리 존재 여부 확인
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
        print(f"'{repo_id}' 리포지토리가 이미 존재합니다. 기존 리포지토리에 데이터셋을 업로드합니다.")
    except Exception as e:
        # 리포지토리가 없으면 생성
        print(f"'{repo_id}' 리포지토리가 존재하지 않습니다. 새로 생성한 후 업로드합니다.")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=token)

    # 폴더 내 데이터 파일 경로 설정
    folder_path = os.path.join("..", "data", foler_name)
    data_files = {}

    # train/test 파일을 각각 데이터셋의 분할로 지정
    if os.path.exists(os.path.join(folder_path, "train.csv")):
        data_files["train"] = os.path.join(folder_path, "train.csv")
    if os.path.exists(os.path.join(folder_path, "test.csv")):
        data_files["test"] = os.path.join(folder_path, "test.csv")

    if not data_files:
        print(f"폴더 '{folder_path}'에 업로드할 유효한 데이터 파일이 없습니다.")
        return

    # 데이터셋 로드 및 업로드
    dataset = load_dataset("csv", data_files=data_files)
    dataset.push_to_hub(repo_id, token=token)
    print(f"데이터셋이 '{repo_id}'에 업로드되었습니다.")


if '__main__' == __name__:
    # 허깅페이스 API키 관리
    load_env_file("../setup/.env")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")


    # folder name 입력
    # origin 입력시 paper-company/datacentric-origin 형태로 저장
    #########################################
    foler_name = 'origin'
    #########################################

    upload_dataset_folder_to_hub(foler_name, hf_token, private=True)
