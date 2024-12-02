import os
import time
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from retriever import HybridSearch
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()


class PromptTemplateGeneration:
    def __init__(self) -> None:
        self.search = HybridSearch()  # HybridSearch 모듈 사용
        self.prompt_str = """
        You are an AI assistant specializing in retrieving and interpreting data from structured information. 
        Your task is to accurately extract relevant information from the provided context and answer the query based on that data.

        Context:
        {context_str}

        Query: {query_str}

        Please follow these guidelines in your response:
        1. Extract the specific data relevant to the query from the context.
        2. Based on the extracted data, provide a clear and concise answer to the query.
        3. If the context does not contain enough information to answer the query, mention that there is insufficient information.
        4. Your response should be in Korean, clear, and easy to understand.

        Response:
        """
        self.prompt_tmpl = PromptTemplate(self.prompt_str)

    def prompt_generation(self, query: str, filename: str) -> str:
        metadata_filter = self.search.metadata_filter(filename)
        results = self.search.query_hybrid_search(query, metadata_filter=None)

        # 검색된 문서 로그 출력
        print("검색된 문서:")
        for result in results:
            print(f"- File Name: {result['file_name']}, Chunk ID: {result['chunk_id']}")

        # 검색 결과를 바탕으로 컨텍스트 생성
        context = "\n\n".join([result["llm_text"] for result in results])
        return self.prompt_tmpl.format(context_str=context, query_str=query)


class RAGStringQueryEngine:
    def __init__(self, model_url: str, api_key: str, max_tokens: int):
        self.chat_llm = ChatOpenAI(
            model=model_url,
            base_url="http://10.10.10.109:11434/v1",  # 모델 서버 URL
            api_key=api_key,
            max_tokens=max_tokens
        )
        self.response_synthesizer = TreeSummarize()

    def custom_query(self, prompt: str) -> str:
        messages = [
            "You are an AI assistant specializing in interpreting structured information.",
            prompt
        ]

        try:
            # 추론 시간 측정 시작
            start_time = time.time()

            # LLM 호출
            response = self.chat_llm.generate(messages=messages)

            # 추론 시간 측정 종료
            elapsed_time = time.time() - start_time
            print(f"추론 시간: {elapsed_time:.2f}초")

            if isinstance(response.generations, list) and response.generations:
                completion_text = response.generations[0][0].text.strip()
            else:
                raise ValueError("Invalid or empty response from the model.")

            # 응답 요약
            summary = self.response_synthesizer.get_response(
                query_str=completion_text,
                text_chunks=[prompt]
            )
            return str(summary)
        except Exception as e:
            raise RuntimeError(f"Error during LLM generation: {e}")


def create_query_engine(prompt: str, model_url: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "your_default_api_key")  # API 키 로드
    max_tokens = 2048
    query_engine = RAGStringQueryEngine(
        model_url=model_url,
        api_key=api_key,
        max_tokens=max_tokens
    )
    return query_engine.custom_query(prompt)


if __name__ == '__main__':
    # 사용할 모델 리스트
    model_list = [
        "hf.co/bartowski/gemma-2-9b-it-GGUF:Q8_0",
        "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0",
        "hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q8_0",
        "hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:latest",
        "hf.co/mradermacher/Llama-3.1-Korean-8B-Instruct-GGUF:Q8_0",
        "hf.co/QuantFactory/ko-gemma-2-9b-it-GGUF:Q8_0",
        "hf.co/teddylee777/Qwen2.5-7B-Instruct-kowiki-qa-context-gguf:Q8_0"
    ]

    print("Available Models:")
    for idx, model in enumerate(model_list):
        print(f"{idx + 1}: {model}")

    # 사용자 입력으로 모델 선택
    model_index = int(input("Choose a model by entering the corresponding number: ")) - 1
    if model_index < 0 or model_index >= len(model_list):
        raise ValueError("Invalid model selection.")
    selected_model = model_list[model_index]

    # 쿼리와 파일명 설정
    # query_str = "모든 사용자의 이름을 나열해줘" ## user
    # query_str = "강희창의 아이디 알려줘" ## user
    # query_str = "강희창의 사용권한 그룹넘버 알려줘" ## user, user_auth_grp
    query_str = "산업보건의의 조직관리 번호는 뭐야?" ## com_manager
    # query_str = "알람번호가 16인 작업의 알람명 알려줘" ## alarm
    # query_str = "개선조치 예정일 초과알림의 알람주기 알려줘" ## alarm, alarm_cycle
    # query_str = "2024년 3월에 등록된 설비 이름과 각 설비의 설비 유형을 알려줘" ## shm_facility, com_facility_type
    
    # query_str = "2024년 3월에 등록된 설비 이름과 각 설비의 설비유형 명을 알려줘" ## shm_facility, com_facility_type
    # query_str = "보건관리자 명단을 알려줘" ## shm_isc_report_attendee
    # query_str = "심용섭이 등록한 모든 정보를 알려줘"
    filename = ""

    # 프롬프트 생성 및 응답 생성
    prompt_gen = PromptTemplateGeneration()
    prompt = prompt_gen.prompt_generation(query=query_str, filename=filename)
    response = create_query_engine(prompt, selected_model)
    print(response)
