# from langchain_openai import ChatOpenAI

# chat_llm = ChatOpenAI(
#     model="hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M",
#     base_url="http://10.10.10.109:11434/v1",
#     api_key="ollama",
#     max_tokens=2048
# )

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "안녕하세요. 어떻게 도와드릴까요?"}
# ]

# try:
#     response = chat_llm(messages=messages)
#     print("Response:", response)
# except Exception as e:
#     print("Connection Error:", str(e))

import requests

url = "http://10.10.10.109:11434/v1"
try:
    response = requests.get(url, timeout=5)
    print("Connection successful:", response.status_code)
except requests.exceptions.RequestException as e:
    print("Connection error:", e)
