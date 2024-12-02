import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from typing import List, Union
import openai
from rerank import reranking  # 리랭크 모듈 사용

# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')
OpenAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = OpenAI_API_KEY

class HybridSearch:
    """
    A class for performing hybrid search using dense and sparse embeddings and applying reranking.
    """

    def __init__(self) -> None:
        """
        Initialize the HybridSearch object with OpenAI's dense embedding model and a sparse embedding model.
        """
        # Sparse embedding model initialization
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        
        # Qdrant client initialization
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

    def get_openai_embedding(self, text: str) -> List[float]:
        """
        Get dense embedding using OpenAI's embedding model.
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response["data"][0]["embedding"]
            return embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return []

    def metadata_filter(self, file_names: Union[str, List[str]]) -> Union[None, models.Filter]:
        """
        Create a metadata filter based on the file names provided.
        If no file_names are provided, return None for no filter.
        """
        if not file_names:
            return None

        if isinstance(file_names, str):
            file_name_condition = models.FieldCondition(
                key="file_name",
                match=models.MatchValue(value=file_names)
            )
        else:
            file_name_condition = models.FieldCondition(
                key="file_name",
                match=models.MatchAny(any=file_names)
            )

        return models.Filter(
            must=[file_name_condition]
        )
    
    def query_hybrid_search(self, query: str, metadata_filter=None, limit=5):
        """
        Perform a hybrid search using dense and sparse embeddings and apply reranking to the results.
        """
        # Generate dense embedding using OpenAI
        dense_query = self.get_openai_embedding(query)
        if not dense_query:
            logger.error("Failed to retrieve dense embedding. Aborting search.")
            return []

        # Generate sparse embedding using the sparse embedding model
        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

        try:
            # Perform hybrid search using query_points
            results = self.qdrant_client.query_points(
                collection_name=Collection_Name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query.indices.tolist(),
                            values=sparse_query.values.tolist(),
                        ),
                        using="sparse",
                        limit=limit,
                    ),
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=limit,
                    ),
                ],
                query_filter=metadata_filter,
                query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
            )

            # Extract the text from the payload of each scored point
            documents = [
                {
                    "llm_text": point.payload.get("llm_text", "No LLM text found"),
                    "file_name": point.payload.get("file_name", "No file name found"),
                    "chunk_id": point.payload.get("chunk_id", "No chunk ID found"),
                }
                for point in results.points
            ]

            # Extract only the 'llm_text' for reranking
            llm_texts = [doc["llm_text"] for doc in documents]

            # Rerank the documents using the reranking model
            reranker = reranking()  # Initialize the reranking module
            top_llm_texts = reranker.rerank_documents(query, llm_texts)

            # Filter the top-ranked documents with their original metadata
            top_documents = [doc for doc in documents if doc["llm_text"] in top_llm_texts]

            return top_documents

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []

if __name__ == '__main__':
    search = HybridSearch()
    
    # Query input
    query = "이름이 '심용섭'인 사람의 그룹넘버 알려줘"
    
    # Metadata filter setup
    # file_names = ["user"]  # Example file names to filter
    # metadata_filter = search.metadata_filter(file_names)
    metadata_filter = None
    

    # Perform hybrid search
    results = search.query_hybrid_search(query, metadata_filter, limit=5)
    logger.info(f"Found {len(results)} results for query: {query}")
    
    # Output results
    for idx, document in enumerate(results, start=1):
        logger.info(f"Result {idx}: File Name: {document['file_name']}, Chunk ID: {document['chunk_id']}, LLM Text: {document['llm_text']}")

# ## 테이블 하나 참조
# # question ="모든 사용자의 이름만 나열해줘"
# # question = "user_nm이 '이보현'인 user_id를 보여줘"
# # question = "이름이 '이보현'인 아이디를 보여줘"
# # question ="비활성화된 계정(use_yn = 'N')의 사용자 수를 계산해줘."
# # question ="비활성화된 계정의 사용자 수를 계산해줘."
# # question ="로그인 아이디가 'sapair'인 사람의 유저 아이디 보여줘"

# ## 테이블 두개 참조
# # question = "이름이 '이보현'인 사람의 user_token_store_no 알려줘"
# # question = "이보현의 토큰번호 알려줘"
# # question = "이름이 '이보현'인 사람의 그룹넘버 알려줘"