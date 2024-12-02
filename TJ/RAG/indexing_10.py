
import logging
from dotenv import load_dotenv
import os
import json
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from pathlib import Path
import openai
from collections import defaultdict
from uuid import uuid4  # UUID를 생성하기 위해 추가

# Load environment variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')
OpenAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = OpenAI_API_KEY

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantIndexing:
    """
    A class for indexing documents using Qdrant vector database.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Initialize the QdrantIndexing object.
        """
        self.data_dir = data_dir
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY
        )
        logging.info("QdrantIndexing object initialized.")

    def load_nodes(self):
        """
        Load nodes from JSON files in the data directory and group them by file_name.
        """
        self.nodes_by_file = defaultdict(list)

        for json_file in Path(self.data_dir).glob("*.json"):
            with open(json_file, 'r') as file:
                node = json.load(file)
                file_name = node['metadata']['file_name']
                self.nodes_by_file[file_name].append(node)

        # Sort each group by chunk_id
        for file_name, nodes in self.nodes_by_file.items():
            self.nodes_by_file[file_name] = sorted(nodes, key=lambda x: x['metadata']['chunk_id'])

        logging.info(f"Loaded nodes grouped by file_name: {len(self.nodes_by_file)} file groups.")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if not self.qdrant_client.collection_exists(collection_name=f"{Collection_Name}"): 
            self.qdrant_client.create_collection(
                collection_name=Collection_Name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=1536,  # OpenAI embedding size for text-embedding-ada-002
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,              
                        ),
                    )
                }
            )
            logging.info(f"Created collection '{Collection_Name}' in Qdrant vector database.")

    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using SPLADE.
        """
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]
        
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")

    def create_dense_vector(self, text):
        """
        Create a dense vector from the text using OpenAI embeddings.
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # OpenAI embedding model
            )
            dense_vector = response["data"][0]["embedding"]
            return dense_vector

        except Exception as e:
            logging.error(f"Error generating OpenAI embedding: {e}")
            return None
        
    def documents_insertion(self):
        """
        Insert documents into the Qdrant vector database with both dense and sparse embeddings.
        """
        points = []
        for file_name, nodes in self.nodes_by_file.items():
            logging.info(f"Processing file: {file_name}")

            # Limit to the first 10 chunks per file
            limited_nodes = nodes[:10]

            for node in tqdm(limited_nodes, total=len(limited_nodes)):
                # Extract texts
                sparse_text = node.get("sparse_text", "")
                dense_text = node.get("dense_text", "")
                llm_text = node.get("llm_text", "")
                metadata = node.get("metadata", {})

                # Generate dense embedding using OpenAI for dense_text
                dense_embedding = self.create_dense_vector(dense_text)
                if dense_embedding is None:
                    continue  # Skip this document if dense embedding generation fails

                # Generate sparse vector using SPLADE for sparse_text
                sparse_vector = self.create_sparse_vector(sparse_text)

                # Create a unique ID using UUID
                unique_id = str(uuid4())  # 고유한 UUID 생성

                # Create PointStruct
                point = models.PointStruct(
                    id=unique_id,  # Use UUID as the point ID
                    vector={
                        'dense': dense_embedding,
                        'sparse': sparse_vector,
                    },
                    payload={
                        'sparse_text': sparse_text,
                        'dense_text': dense_text,
                        'llm_text': llm_text,
                        **metadata  # Include all metadata
                    }
                )
                points.append(point)

        # Upsert points into Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=Collection_Name,
                points=points
            )
            logging.info(f"Upserted {len(points)} points with dense and sparse vectors into Qdrant vector database.")
        else:
            logging.error("No points to insert into Qdrant.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # indexing = QdrantIndexing(data_dir="./processed5_json")
    indexing = QdrantIndexing(data_dir="./processed_korean_2_json")
    indexing.load_nodes()
    indexing.client_collection()
    indexing.documents_insertion()
