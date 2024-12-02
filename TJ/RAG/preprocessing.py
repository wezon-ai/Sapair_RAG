import pandas as pd
import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime

# 텍스트 전처리 함수
def preprocess_text(text):
    text = str(text).strip().lower()  # 소문자 변환 및 공백 제거
    text = re.sub(r'[^\w\s|]', '', text)  # 특수 문자 제거 ('|' 기호 제외)
    text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
    return text

# CSV 파일을 원본 및 전처리된 텍스트로 변환하는 함수
def csv_to_chunk_with_raw(input_csv):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv, encoding='utf-8')

    # 원본 데이터를 마크다운 형식으로 병합 (raw_text)
    raw_markdown_lines = []
    for _, row in df.iterrows():
        row_text = "| " + " | ".join(map(str, row.values)) + " |"
        raw_markdown_lines.append(row_text)
    raw_text = "\n".join(raw_markdown_lines)  # 원본 텍스트

    # 전처리된 데이터프레임 생성 및 마크다운 변환
    df_processed = df.applymap(preprocess_text)
    processed_markdown_lines = []
    for _, row in df_processed.iterrows():
        row_text = "| " + " | ".join(map(str, row.values)) + " |"
        processed_markdown_lines.append(row_text)
    processed_text = "\n".join(processed_markdown_lines)  # 전처리된 텍스트

    return raw_text, processed_text, df_processed

# JSON 형식으로 저장하는 함수
def save_to_json(output_dir, schema_name, file_name, raw_text, processed_text, df, file_path):
    # 추가 메타정보 계산
    created_at = datetime.now().isoformat()  # 현재 시간
    processed_at = datetime.now().isoformat()
    num_rows, num_columns = df.shape  # 행과 열 개수
    file_size_kb = os.path.getsize(file_path) / 1024  # 파일 크기 (KB)

    # JSON 데이터 생성
    json_data = {
        "id_": str(uuid.uuid4()),  # 고유 ID 생성
        "embedding": None,
        "metadata": {
            "schema": schema_name,  # 스키마 이름
            "file_name": file_name,  # 파일 이름
            "created_at": created_at,
            "processed_at": processed_at,
            "num_rows": num_rows,
            "num_columns": num_columns,
            "file_size_kb": round(file_size_kb, 2)  # 소수점 2자리까지
        },
        "raw_text": raw_text,  # 원본 텍스트
        "processed_text": processed_text,  # 전처리된 텍스트
        "mimetype": "text/plain"
    }

    # JSON 파일 저장
    output_path = os.path.join(output_dir, f"{file_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"Saved JSON: {output_path}")

# 메인 실행 함수
def process_all_csv(input_dir, output_dir, schema_name):
    for csv_file in Path(input_dir).glob("*.csv"):
        print(f"Processing: {csv_file.name}")
        
        # CSV를 raw_text와 processed_text로 변환
        raw_text, processed_text, df_processed = csv_to_chunk_with_raw(csv_file)
        
        # JSON 파일로 저장
        save_to_json(output_dir, schema_name, csv_file.stem, raw_text, processed_text, df_processed, csv_file)

# 실행 코드
if __name__ == "__main__":
    # 입력 CSV 디렉토리와 출력 JSON 디렉토리
    input_dir = "./wezontest_csv"  # CSV 파일이 있는 디렉토리
    output_dir = "./processed3_json"  # JSON 파일을 저장할 디렉토리
    schema_name = "wezontest"  # 스키마 이름

    # 모든 CSV 처리 및 JSON 저장
    process_all_csv(input_dir, output_dir, schema_name)
