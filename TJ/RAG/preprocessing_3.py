import pandas as pd
import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime


# Dense Text 전처리 함수 (컬럼-값 매칭 포함)
def preprocess_dense_text_with_columns(row, column_names):
    """
    Dense Text용 행 데이터를 컬럼 이름과 매칭하여 텍스트로 변환합니다.
    """
    row_data = []
    for col_name, value in zip(column_names, row):
        # 의미 없는 값 제거 및 기본 전처리
        value = str(value).strip()
        if value.lower() in ['nan', 'null', '', 'none']:
            value = ""
        row_data.append(f"{col_name}: {value}")
    return ", ".join(row_data)


# Sparse Text 전처리 함수
def preprocess_sparse_text(text):
    """
    Sparse Text 전처리: 소문자 변환, 특수 문자 제거, 연속 공백 제거
    """
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
    return text


# LLM Text 전처리 함수 (컬럼-값 매칭 포함)
def preprocess_llm_text_with_columns(row, column_names):
    """
    LLM Text용 행 데이터를 컬럼 이름과 매칭하여 텍스트로 변환합니다.
    """
    row_data = []
    for col_name, value in zip(column_names, row):
        row_data.append(f"{col_name}: {value}")
    return ", ".join(row_data)


# CSV 파일을 Sparse, Dense, LLM 텍스트로 변환하는 함수 (청크 생성)
def csv_to_chunks(input_csv, max_chunk_length=400):
    """
    CSV 파일을 읽어 Sparse, Dense, LLM 텍스트로 변환하고 청크를 생성합니다.
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_csv, encoding='utf-8')

    # 컬럼 이름 저장
    column_names = df.columns.tolist()

    # Sparse, Dense, LLM 텍스트 초기화
    sparse_chunks = []
    dense_chunks = []
    llm_chunks = []

    current_sparse_chunk = []
    current_dense_chunk = []
    current_llm_chunk = []

    current_chunk_length = 0

    for _, row in df.iterrows():
        # 각 행의 데이터를 전처리
        row_sparse = preprocess_sparse_text("| " + " | ".join(map(str, row.values)) + " |")
        row_dense = preprocess_dense_text_with_columns(row, column_names)
        row_llm = preprocess_llm_text_with_columns(row, column_names)

        # 유효한 행만 추가 (빈 텍스트 제외)
        if not row_dense.strip():
            continue

        row_length = len(row_llm)  # LLM Text를 기준으로 길이 측정

        # 현재 청크에 데이터 추가
        if current_chunk_length + row_length > max_chunk_length:
            # 현재 청크 저장
            sparse_chunks.append("\n".join(current_sparse_chunk))
            dense_chunks.append("\n".join(current_dense_chunk))
            llm_chunks.append("\n".join(current_llm_chunk))

            # 새로운 청크 시작
            current_sparse_chunk = [row_sparse]
            current_dense_chunk = [row_dense]
            current_llm_chunk = [row_llm]
            current_chunk_length = row_length
        else:
            current_sparse_chunk.append(row_sparse)
            current_dense_chunk.append(row_dense)
            current_llm_chunk.append(row_llm)
            current_chunk_length += row_length

    # 마지막 청크 저장
    if current_sparse_chunk:
        sparse_chunks.append("\n".join(current_sparse_chunk))
        dense_chunks.append("\n".join(current_dense_chunk))
        llm_chunks.append("\n".join(current_llm_chunk))

    return sparse_chunks, dense_chunks, llm_chunks, df


# JSON 형식으로 저장하는 함수
def save_to_json(output_dir, schema_name, file_name, sparse_chunks, dense_chunks, llm_chunks, df, file_path):
    """
    전처리된 데이터를 JSON 형식으로 저장합니다.
    """
    # 추가 메타정보 계산
    created_at = datetime.now().isoformat()  # 현재 시간
    processed_at = datetime.now().isoformat()
    num_rows, num_columns = df.shape  # 행과 열 개수
    file_size_kb = os.path.getsize(file_path) / 1024  # 파일 크기 (KB)

    # JSON 데이터 생성
    for i, (sparse_text, dense_text, llm_text) in enumerate(zip(sparse_chunks, dense_chunks, llm_chunks)):
        json_data = {
            "id_": str(uuid.uuid4()),  # 고유 ID 생성
            "embedding": None,
            "metadata": {
                "schema": schema_name,  # 스키마 이름
                "file_name": file_name,  # 파일 이름
                "chunk_id": i,  # 청크 ID
                "created_at": created_at,
                "processed_at": processed_at,
                "num_rows": num_rows,
                "num_columns": num_columns,
                "file_size_kb": round(file_size_kb, 2)  # 소수점 2자리까지
            },
            "sparse_text": sparse_text,  # 희소 임베딩용 텍스트
            "dense_text": dense_text,  # 밀집 임베딩용 텍스트
            "llm_text": llm_text,  # LLM 프롬프트용 원본 텍스트
            "mimetype": "text/plain"
        }

        # JSON 파일 저장
        output_path = os.path.join(output_dir, f"{file_name}_chunk_{i}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON: {output_path}")


# 메인 실행 함수
def process_all_csv(input_dir, output_dir, schema_name):
    """
    입력 디렉토리 내 모든 CSV 파일을 처리합니다.
    """
    for csv_file in Path(input_dir).glob("*.csv"):
        print(f"Processing: {csv_file.name}")

        # CSV를 Sparse, Dense, LLM 텍스트로 변환
        sparse_chunks, dense_chunks, llm_chunks, df = csv_to_chunks(csv_file)

        # JSON 파일로 저장
        save_to_json(output_dir, schema_name, csv_file.stem, sparse_chunks, dense_chunks, llm_chunks, df, csv_file)


# 실행 코드
if __name__ == "__main__":
    # 입력 CSV 디렉토리와 출력 JSON 디렉토리
    input_dir = "./wezontest_csv"  # CSV 파일이 있는 디렉토리
    output_dir = "./processed4_json"  # JSON 파일을 저장할 디렉토리
    schema_name = "wezontest"  # 스키마 이름

    # 모든 CSV 처리 및 JSON 저장
    process_all_csv(input_dir, output_dir, schema_name)
