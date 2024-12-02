## rerank.py

from sentence_transformers import CrossEncoder

class reranking():
    def __init__(self) -> None:
        # Load the CrossEncoder model
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank_documents(self, query, documents):
        # print(f"Received Query: {query}")  # 쿼리 확인
        # print(f"Documents for Reranking: {documents}")  # 문서 확인
        
        # Compute the similarity scores between the query and each document
        scores = self.model.predict([(query, doc) for doc in documents])
        print(f"Scores: {scores}")  # 유사도 점수 확인

        # Sort the documents based on their similarity scores
        ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # Select the top 2 documents
        top_documents = [doc for doc, score in ranked_documents[:3]]
        # print(f"Top Documents: {top_documents}")  # 상위 2개의 문서 확인

        return top_documents
