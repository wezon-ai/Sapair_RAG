## generation.py

import openai
import os
import time  # 시간 측정을 위한 모듈 추가
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from retriever import HybridSearch

# Load environment variables (including OpenAI API key)
load_dotenv()

## 1. prompt_template_generation 클래스
class prompt_template_generation():
    def __init__(self) -> None:
        self.search = HybridSearch()  # 수정된 하이브리드 검색 모듈 사용
        # 프롬프트 템플릿 수정
        self.prompt_str = """You are an AI assistant specializing in retrieving and interpreting data from structured information. Your task is to accurately extract relevant information from the provided context and answer the query based on that data.

        Context:
        {context_str}

        Query: {query_str}

        Please follow these guidelines in your response:
        1. Extract the specific data relevant to the query from the context.
        2. Based on the extracted data, provide a clear and concise answer to the query.
        3. If the context does not contain enough information to answer the query, mention that there is insufficient information and respond based only on the provided context.
        4. Your response should be in Korean, clear, and easy to understand.

        Response:
        """
        self.prompt_tmpl = PromptTemplate(self.prompt_str)

    def prompt_generation(self, query: str, filename: str):
        # 메타데이터 필터 설정
        metadata_filter = self.search.metadata_filter(filename)
        # 하이브리드 검색 수행
        results = self.search.query_hybrid_search(query, metadata_filter=None)

        # 결과를 기반으로 컨텍스트 생성
        context = "\n\n".join([result["llm_text"] for result in results])

        # 프롬프트 생성
        prompt_templ = self.prompt_tmpl.format(context_str=context, query_str=query)
        print("Generated prompt:", prompt_templ)  # 터미널에 로그 출력

        return prompt_templ

## 2. RAGStringQueryEngine 클래스
class RAGStringQueryEngine(CustomQueryEngine):
    api_key: str
    response_synthesizer: TreeSummarize

    def __init__(self, api_key: str, response_synthesizer: TreeSummarize):
        # API 키 및 응답 합성기 초기화
        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'response_synthesizer', response_synthesizer)

    def custom_query(self, prompt: str) -> str:
        openai.api_key = self.api_key
        print(f"Using API Key: {openai.api_key}")  # API 키 로그 출력

        # 추론 시간 측정 시작
        start_time = time.time()
        
        # GPT-4 API 호출
        response = openai.chat.completions.create(
            model="gpt-4o",  # 최신 GPT 모델 사용
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in interpreting structured information."},
                {"role": "user", "content": prompt}
            ],
        )

        # 추론 시간 측정 종료
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference Time: {inference_time:.2f} seconds")  # 추론 시간 출력

        # 응답 확인
        print("OpenAI API response:", response)  # 터미널에 로그 출력
        completion_text = response.choices[0].message.content.strip()

        # TreeSummarize를 사용한 응답 요약
        summary = self.response_synthesizer.get_response(query_str=completion_text, text_chunks=prompt)
        return str(summary)

def create_query_engine(prompt: str):
    api_key = os.environ.get('OPENAI_API_KEY')  # OpenAI API 키를 환경변수에서 불러옴
    response_synthesizer = TreeSummarize()  # TreeSummarize 응답 합성기

    query_engine = RAGStringQueryEngine(
        api_key=api_key,
        response_synthesizer=response_synthesizer
    )
    
    response = query_engine.custom_query(prompt)
    return response


if __name__ == '__main__':
    # 쿼리와 파일명 입력
     # 쿼리와 파일명 입력
    # query_str = "모든 사용자의 이름을 나열해줘" ## user
    # query_str = "강희창의 아이디 알려줘" ## user
    # query_str = "강희창의 사용권한 그룹넘버 알려줘" ## user, user_auth_grp
    # query_str = "산업보건의의 조직관리 번호는 뭐야?" ## com_manager
    # query_str = "알람번호가 16인 작업의 알람명 알려줘" ## alarm
    # query_str = "개선조치 예정일 초과알림의 알람주기 알려줘" ## alarm, alarm_cycle
    # query_str = "2024년 3월에 등록된 설비 이름과 각 설비의 설비 유형을 알려줘" ## shm_facility, com_facility_type
    
    # query_str = "2024년 3월에 등록된 설비 이름과 각 설비의 설비유형 명을 알려줘" ## shm_facility, com_facility_type
    query_str = "보건관리자 명단을 알려줘" ## shm_isc_report_attendee
    # query_str = "심용섭이 등록한 모든 정보를 알려줘"
    filename = ""

    # 프롬프트 생성 및 응답 생성
    prompt_gen = prompt_template_generation()
    prompt = prompt_gen.prompt_generation(query=query_str, filename=filename)
    response = create_query_engine(prompt)
    print(response)
