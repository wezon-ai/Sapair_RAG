import pandas as pd
import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime


# 엑셀 파일에서 매핑 데이터를 불러오는 함수
def load_mappings(column_mapping_path, table_mapping_path):
    """
    엑셀 파일에서 컬럼 이름과 테이블 이름 매핑을 로드합니다.
    """
    column_mappings = pd.read_excel(column_mapping_path, engine="openpyxl")
    table_mappings = pd.read_excel(table_mapping_path, engine="openpyxl")
    column_name_dict = dict(zip(column_mappings["컬럼명"], column_mappings["한글 컬럼명"]))
    table_name_dict = dict(zip(table_mappings["테이블명"], table_mappings["한글 테이블명"]))
    return column_name_dict, table_name_dict


# Dense Text 전처리 함수 (한글 컬럼명 적용)
def preprocess_dense_text_with_columns_korean(row, column_names, column_name_dict):
    row_data = []
    for col_name, value in zip(column_names, row):
        korean_col_name = column_name_dict.get(col_name, col_name)  # 한글 컬럼명 매핑
        value = str(value).strip()
        if value.lower() in ['nan', 'null', '', 'none']:
            value = ""
        row_data.append(f"{korean_col_name}: {value.lower()}")
    return ", ".join(row_data)


# Sparse Text 전처리 함수
def preprocess_sparse_text(text):
    """
    Sparse Text 전처리: 소문자 변환, 특수 문자 제거, 연속 공백 제거, 'nan' 값 제거
    """
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'\b(nan|null|none)\b', '', text)  # nan, null, none 제거
    text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거
    return text.strip()


# LLM Text 전처리 함수 (한글 컬럼명 적용)
def preprocess_llm_text_with_columns_korean(row, column_names, column_name_dict):
    row_data = []
    for col_name, value in zip(column_names, row):
        korean_col_name = column_name_dict.get(col_name, col_name)
        row_data.append(f"{korean_col_name}: {value}")
    return ", ".join(row_data)


# CSV 파일을 Sparse, Dense, LLM 텍스트로 변환하는 함수
def csv_to_chunks_with_korean(input_csv, column_name_dict, max_chunk_length=400):
    df = pd.read_csv(input_csv, encoding="utf-8")
    column_names = df.columns.tolist()

    sparse_chunks, dense_chunks, llm_chunks = [], [], []
    current_sparse_chunk, current_dense_chunk, current_llm_chunk = [], [], []
    current_chunk_length = 0

    for _, row in df.iterrows():
        row_sparse = preprocess_sparse_text("| " + " | ".join(map(str, row.values)) + " |")  # Sparse Text 생성
        row_dense = preprocess_dense_text_with_columns_korean(row, column_names, column_name_dict)  # Dense Text 생성
        row_llm = preprocess_llm_text_with_columns_korean(row, column_names, column_name_dict)  # LLM Text 생성

        if not row_dense.strip():
            continue

        row_length = len(row_llm)
        if current_chunk_length + row_length > max_chunk_length:
            sparse_chunks.append("\n".join(current_sparse_chunk))
            dense_chunks.append("\n".join(current_dense_chunk))
            llm_chunks.append("\n".join(current_llm_chunk))

            current_sparse_chunk, current_dense_chunk, current_llm_chunk = [row_sparse], [row_dense], [row_llm]
            current_chunk_length = row_length
        else:
            current_sparse_chunk.append(row_sparse)
            current_dense_chunk.append(row_dense)
            current_llm_chunk.append(row_llm)
            current_chunk_length += row_length

    if current_sparse_chunk:
        sparse_chunks.append("\n".join(current_sparse_chunk))
        dense_chunks.append("\n".join(current_dense_chunk))
        llm_chunks.append("\n".join(current_llm_chunk))

    return sparse_chunks, dense_chunks, llm_chunks, df


# JSON 저장 함수
def save_to_json_with_korean(output_dir, schema_name, file_name, sparse_chunks, dense_chunks, llm_chunks, df, file_path, table_name_dict):
    created_at = datetime.now().isoformat()
    processed_at = datetime.now().isoformat()
    num_rows, num_columns = df.shape
    file_size_kb = os.path.getsize(file_path) / 1024

    korean_file_name = table_name_dict.get(file_name, file_name)  # 한글 테이블명 매핑

    for i, (sparse_text, dense_text, llm_text) in enumerate(zip(sparse_chunks, dense_chunks, llm_chunks)):
        sparse_text = f"[file_name: {korean_file_name}, chunk_id: {i}]\n{sparse_text}"
        dense_text = f"[file_name: {korean_file_name}, chunk_id: {i}]\n{dense_text}"
        llm_text = f"[file_name: {korean_file_name}, chunk_id: {i}]\n{llm_text}"

        json_data = {
            "id_": str(uuid.uuid4()),
            "embedding": None,
            "metadata": {
                "schema": schema_name,
                "file_name": korean_file_name,
                "chunk_id": i,
                "created_at": created_at,
                "processed_at": processed_at,
                "num_rows": num_rows,
                "num_columns": num_columns,
                "file_size_kb": round(file_size_kb, 2),
            },
            "sparse_text": sparse_text,
            "dense_text": dense_text,
            "llm_text": llm_text,
            "mimetype": "text/plain",
        }

        output_path = os.path.join(output_dir, f"{file_name}_chunk_{i}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON: {output_path}")


# 메인 실행 함수
def process_all_csv_with_korean(input_dir, output_dir, schema_name, column_mapping_path, table_mapping_path):
    column_name_dict, table_name_dict = load_mappings(column_mapping_path, table_mapping_path)

    for csv_file in Path(input_dir).glob("*.csv"):
        print(f"Processing: {csv_file.name}")
        sparse_chunks, dense_chunks, llm_chunks, df = csv_to_chunks_with_korean(csv_file, column_name_dict)
        save_to_json_with_korean(output_dir, schema_name, csv_file.stem, sparse_chunks, dense_chunks, llm_chunks, df, csv_file, table_name_dict)


# 실행 코드
if __name__ == "__main__":
    input_dir = "./wezontest_csv"
    output_dir = "./processed_korean_json"
    schema_name = "wezontest"
    column_mapping_path = "./column list.xlsx"
    table_mapping_path = "./table list.xlsx"

    process_all_csv_with_korean(input_dir, output_dir, schema_name, column_mapping_path, table_mapping_path)
