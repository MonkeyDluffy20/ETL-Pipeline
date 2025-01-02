"""
This module provides a wrapper around Pinecone's vector database operations, integrating it with OpenAI embeddings. 
It enables seamless creation, querying, updating, and deletion of vector indices for efficient similarity searches.

Classes:
    PineConeClient: 
        A client for managing Pinecone vector databases, supporting operations like index creation, 
        document addition, querying, vector updates, and index deletion.

Functions:
    get_openai_api_key():
        Retrieves the OpenAI API key from environment variables or prompts the user to input it.

Main Features:
    - Initialize Pinecone and OpenAI embeddings with customizable configurations.
    - Create and manage vector indices with options for server type and namespace.
    - Add data to the vector database with embeddings generated using OpenAI models.
    - Query vectors for similarity search based on embeddings.
    - Update, delete, and manage specific vectors or indices.
    - Full logging for successful operations and error handling for robustness.

Usage Example:
    >>> # Initialize OpenAI and Pinecone API keys
    >>> api_key = get_openai_api_key()
    >>> client = PineConeClient(api_key=api_key, index_name="example_index")

    >>> # Create a Pinecone index
    >>> client.create_index(server_type="serverless")

    >>> # Add documents to the index
    >>> from langchain_core.documents import Document
    >>> documents = [Document(page_content="I Like Pinecone",metadata={"source": "tweet"})]
    >>> ids = ["doc1"]
    >>> client.add_data_to_vector_database(documents, ids, namespace="example-ns")base(documents, ids, namespace="example_namespace")

    >>> # Query the index for similar documents
    >>> results = client.query_vector_store(query="What is Pinecone?", k=3)

    >>> # Update a vector in the index
    >>> response = client.update_vector_by_id(
    ...    id="doc1",
    ...    values=[0.1] * pinecone_client.dimension,
    ...    set_metadata={"updated": True},
    ...    namespace="example-ns"
    ... )

    >>> # Delete an index
    >>> client.delete_index()

Dependencies:
    - pinecone: For interaction with Pinecone vector database.
    - langchain: For OpenAI embedding integrations.
    - dotenv: To load API keys and environment variables.
    - logging: For tracking operations and debugging.
    - getpass: For secure user input of sensitive API keys.

"""

import os
import logging
import time
from uuid import uuid4
from typing import Generator, List, Dict, Any, Optional, Union
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec, PodSpec, PineconeApiException
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(override=True)


class HuggingFaceEmbeddingLoader:
    EMBEDDING_LOG_MSG = "Using HuggingFace embeddings"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_CACHED_MODEL = "./cached_model"


class PineconeClientError(Exception):
    """Base exception class for PineconeClient errors."""

    pass


class ConfigurationError(PineconeClientError):
    """Raised when there are configuration or initialization errors."""

    pass


class IndexOperationError(PineconeClientError):
    """Raised when index operations (create, delete, etc.) fail."""

    pass


class VectorOperationError(PineconeClientError):
    """Raised when vector operations (add, update, delete) fail."""

    pass


class QueryError(PineconeClientError):
    """Raised when query operations fail."""

    pass


class EmbeddingError(PineconeClientError):
    """Raised when embedding operations fail."""

    pass


class PineConeClient:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        metric: str = "dotproduct",
        embeddings: str = "openai",
        deletion_protection: str = "disabled",
    ):
        """
        Initialize the PineConeClient.

        Args:
            api_key (str): API key for Pinecone.
            index_name (str): Name of the Pinecone index.
            dimension (int, optional): Dimensionality of embeddings. Defaults to 1536.
            metric (str, optional): Similarity metric to use (e.g., "dotproduct"). Defaults to "dotproduct".
            embeddings (str, optional): Embeddings function to use. Defaults to openai. Choose 'openai' or 'huggingface'.
            deletion_protection (str, optional): Deletion protection for the index. Defaults to "disabled".

        Example:
            >>> client = PineConeClient(api_key="your-api-key", index_name="example-index")
        """

        """Initialize the PineConeClient with error handling for configuration."""
        if not api_key:
            raise ConfigurationError("API key cannot be empty")
        if not index_name:
            raise ConfigurationError("Index name cannot be empty")
        if dimension <= 0:
            raise ConfigurationError("Dimension must be a positive integer")

        try:
            self.api_key = api_key
            self.index_name = index_name
            self.dimension = dimension
            self.metric = metric
            self.pinecone = Pinecone(api_key=api_key)
            self.embeddings = embeddings
            self.deletion_protection = deletion_protection
            self.vector_store = None
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Pinecone client: {str(e)}")

    @staticmethod
    def _validate_server_type(server_type: str) -> None:
        """Validate server type configuration."""
        valid_types = ["pod", "serverless"]
        if server_type not in valid_types:
            raise ConfigurationError(
                f"Invalid server type. Must be one of: {', '.join(valid_types)}"
            )

    def create_index(self, server_type: str) -> None:
        """
        Create a Pinecone index if it does not already exist.

        Args:
            server_type (str): Type of Pinecone server to use ("pod" or "serverless").
            namespace (Optional[str], optional): Optional namespace for the index. Defaults to None.

        Example:
            >>> client.create_index(server_type="serverless")

        Raises:
            ValueError: If the server type is invalid.
            Exception: If the index creation fails.
        """
        try:
            self._validate_server_type(server_type)

            if server_type == "pod":
                environment = os.getenv("ENVIRONMENT", "us-west1-gcp")
                pod_type = os.getenv("POD_TYPE", "p1.x1")
                if not all([environment, pod_type]):
                    raise ConfigurationError("Missing required pod configuration")
                spec = PodSpec(environment=environment, pod_type=pod_type)
            else:  # serverless
                cloud = os.getenv("CLOUD", "aws")
                region = os.getenv("REGION", "us-east-1")
                if not all([cloud, region]):
                    raise ConfigurationError(
                        "Missing required serverless configuration"
                    )
                spec = ServerlessSpec(cloud=cloud, region=region)

            if self.index_name not in [
                idx.name for idx in self.pinecone.list_indexes()
            ]:
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=spec,
                    deletion_protection=self.deletion_protection,
                )
                _logger.info("Waiting for the index to be ready...")
                while not self.pinecone.describe_index(self.index_name).status.get(
                    "ready", False
                ):
                    time.sleep(1)
                _logger.info("Index created successfully.")
            else:
                _logger.info("Index '%s' already exists.", self.index_name)

        except Exception as e:
            _logger.error("Failed to create index: %s", e)
            raise IndexOperationError(f"Failed to create index: {str(e)}")

    # Initialize OpenAI client
    def _get_embeddings(self):
        """
        Retrieve the embedding function based on the specified type.

        Args:
            self: The PineConeClient instance.
        Returns:
            Any: The embedding function object based on the specified type.

        Raises:
            RuntimeError: If the API key is required and not provided for OpenAI.
            ValueError: If an unsupported embedding type is specified.
        """
        try:
            embedding_type = self.embeddings
            if embedding_type == "openai":
                try:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError("API key is required.")
                except Exception as e:
                    _logger.error("Error retrieving API key: %s", e)
                    raise EmbeddingError(f"Failed to get OpenAI API key: {str(e)}")

                model = os.getenv("MODEL", "text-embedding-3-small")
                return OpenAIEmbeddings(model=model)

            elif embedding_type == "huggingface":
                _logger.info(HuggingFaceEmbeddingLoader.EMBEDDING_LOG_MSG)
                return HuggingFaceEmbeddings(
                    model_name=HuggingFaceEmbeddingLoader.HUGGINGFACE_EMBEDDING_MODEL,
                    cache_folder=HuggingFaceEmbeddingLoader.HUGGINGFACE_CACHED_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": False},
                )

            raise EmbeddingError(f"Unsupported embedding type: {embedding_type}")

        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embeddings: {str(e)}")

    def add_data_to_vector_database(
        self,
        documents: List[Dict[str, Any]],
        uuids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Add documents and their IDs to the Pinecone vector database.

        Args:
            documents (List[Dict[str, Any]]): List of documents to add.
            uuids (List[str]): List of IDs corresponding to the documents.
            namespace (str): Namespace for the data.
            embeddings (Any): Embeddings function that will be used.

        Example:
            >>> from langchain_core.documents import Document
            >>> documents = [Document(page_content="I Like Pinecone",metadata={"source": "tweet"})]
            >>> ids = ["doc1"]
            >>> client.add_data_to_vector_database(documents, ids, namespace="example-ns",embeddings=embeddings)

        Raises:
            Exception: If adding data fails.
        """
        try:
            if not documents:
                raise VectorOperationError("No documents provided")
            if not uuids:
                raise VectorOperationError("No UUIDs provided")
            if len(documents) != len(uuids):
                raise VectorOperationError("Number of documents and UUIDs must match")

            index = self.pinecone.Index(self.index_name)
            embeddings = self._get_embeddings()
            self.vector_store = PineconeVectorStore(
                index=index, embedding=embeddings, namespace=namespace
            )
            self.vector_store.add_documents(documents=documents, ids=uuids)
            _logger.info("Documents added to vector database successfully.")
        except PineconeApiException as e:
            _logger.error("Failed to add data to vector database: %s", e)
            raise VectorOperationError(f"Pinecone API error: {str(e)}")
        except Exception as e:
            _logger.error("Failed to add data to vector database: %s", e)
            raise VectorOperationError(f"Failed to add data: {str(e)}")

    def query_vector_store(
        self, query: str, k: int = 5, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone vector database.

        Args:
            query (str): Query string to search for.
            k (int, optional): Number of top results to return. Defaults to 5.
            namespace (Optional[str], optional): Optional namespace for filtering the query. Defaults to None.
        Returns:
            List[Dict[str, Any]]: List of query results.

        Example:
            >>> results = client.query_vector_store(query="Pinecone", k=3, namespace="example-ns")

        Raises:
            Exception: If the query fails.
        """
        try:
            if not query.strip():
                raise QueryError("Query string cannot be empty")
            if k <= 0:
                raise QueryError("Number of results (k) must be positive")
            embeddings = self._get_embeddings()
            index = self.pinecone.Index(self.index_name)
            self.vector_store = PineconeVectorStore(
                index=index, embedding=embeddings, namespace=namespace
            )
            results = self.vector_store.similarity_search(
                query=query, k=k, namespace=namespace
            )
            _logger.info("Query executed successfully.")
            return results
        except PineconeApiException as e:
            _logger.error("Pinecone API error during query: %s", e)
            raise QueryError(f"Pinecone API error during query: {str(e)}")
        except Exception as e:
            _logger.error("Query failed: %s", e)
            raise QueryError(f"Query failed: {str(e)}")

    def delete_data_by_id(
        self, ids: List[str], namespace: Optional[str] = None
    ) -> None:
        """
        Delete documents from the vector database by their IDs.

        Args:
            ids (List[str]): List of document IDs to delete.
            namespace (Optional[str], optional): Optional namespace for the IDs. Defaults to None.

        Example:
            >>> client.delete_data_by_id(ids=["doc1"], namespace="example-ns")

        Raises:
            Exception: If the deletion fails.
        """
        try:
            if not ids:
                raise VectorOperationError("No IDs provided for deletion")
            embeddings = self._get_embeddings()
            index = self.pinecone.Index(self.index_name)
            self.vector_store = PineconeVectorStore(
                index=index, embedding=embeddings, namespace=namespace
            )
            if self.vector_store is None:
                raise VectorOperationError("Vector store is not initialized")
            self.vector_store.delete(ids=ids, namespace=namespace)
            _logger.info("Data deleted successfully.")
        except PineconeApiException as e:
            _logger.error("Pinecone API error during deletion: %s", e)
            raise VectorOperationError(f"Pinecone API error during deletion: {str(e)}")
        except Exception as e:
            _logger.error("Failed to delete data by ID: %s", e)
            raise VectorOperationError(f"Failed to delete data: {str(e)}")

    def list_indexes(self) -> List[str]:
        """
        List all Pinecone indexes.

        Returns:
            List[str]: List of index names.

        Example:
            >>> indexes = client.list_indexes()

        Raises:
            Exception: If listing indexes fails.
        """
        try:
            indexes = [idx.name for idx in self.pinecone.list_indexes()]
            _logger.info("Indexes: %s", indexes)
            return indexes
        except PineconeApiException as e:
            _logger.error("Pinecone API error while listing indexes: %s", e)
            raise IndexOperationError(
                f"Pinecone API error while listing indexes: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to list indexes: %s", e)
            raise IndexOperationError(f"Failed to list indexes: {str(e)}")

    def describe_index(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Describes a Pinecone index.

        Args:
            index_name (Optional[str], optional): Name of the index to describe.
                Defaults to the client's index name if not provided.

        Returns:
            Dict[str, Any]: A dictionary containing the details of the specified index.

        Raises:
            Exception: If describing the index fails.

        Example:
            >>> index_info = client.describe_index(index_name="my_index")
        """
        try:
            index_name = index_name or self.index_name
            if not index_name:
                raise IndexOperationError("No index name provided")

            description = self.pinecone.describe_index(index_name)
            _logger.info(f"Successfully retrieved description for index '{index_name}'")
            return description

        except PineconeApiException as e:
            _logger.info("Pinecone API error while describing index: %s", e)
            raise IndexOperationError(
                f"Pinecone API error while describing index: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to describe index: %s", e)
            raise IndexOperationError(f"Failed to describe index: {str(e)}")

    def delete_index(self, index_name: Optional[str] = None) -> None:
        """
        Deletes a Pinecone index.

        Args:
            index_name (Optional[str], optional): Name of the index to delete.
                Defaults to the client's index name if not provided.

        Returns:
            None

        Raises:
            Exception: If deleting the index fails.

        Example:
            >>> client.delete_index(index_name="my_index")
        """
        try:
            index_name = index_name or self.index_name
            if not index_name:
                raise IndexOperationError("No index name provided")

            self.pinecone.delete_index(index_name)
            _logger.info(f"Successfully deleted index '{index_name}'")

        except PineconeApiException as e:
            _logger.error("Pinecone API error while deleting index: %s", e)
            raise IndexOperationError(
                f"Pinecone API error while deleting index: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to delete index: %s", e)
            raise IndexOperationError(f"Failed to delete index: {str(e)}")

    def _create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text using the specified embedding function.

        Args:

            text (str): The text to embed.

        Returns:
            List[float]: The embedding values for the text.
        """
        return self._get_embeddings().embed_query(text)

    def update_vector_by_id(
        self,
        id: str,
        text: List[float],
        set_metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Updates a vector in the Pinecone database using its ID.

        Args:
            id (str): The unique identifier of the vector to update.
            values (List[float]): The new embedding values for the vector.
            set_metadata (Optional[Dict[str, Any]], optional): Metadata to associate with the vector.
                Defaults to None.
            namespace (Optional[str], optional): Optional namespace for the vector.
                Defaults to None.

        Returns:
            Union[Dict[str, Any], str]: The update response if successful, or an error message.

        Raises:
            Exception: If updating the vector fails.

        Example:
            >>> response = client.update_vector_by_id(
            ...    id="doc1",
            ...    values=[0.1] * pinecone_client.dimension,
            ...    set_metadata={"updated": True},
            ...    namespace="example-ns"
            ... )
        """
        try:
            if not id:
                raise VectorOperationError("Vector ID cannot be empty")
            if not text:
                raise VectorOperationError("Text values cannot be empty")

            index = self.pinecone.Index(self.index_name)

            values = self._create_embedding(text)
            update_response = index.update(
                id=id, values=values, set_metadata=set_metadata, namespace=namespace
            )
            _logger.info(f"Successfully updated vector '{id}'")
            return update_response
        except PineconeApiException as e:
            _logger.error("Pinecone API error during vector update: %s", e)
            raise VectorOperationError(
                f"Pinecone API error during vector update: {str(e)}"
            )
        except Exception as e:
            _logger.error("Failed to update vector: %s", e)
            raise VectorOperationError(f"Failed to update vector: {str(e)}")

    def get_namespaces_detail(self) -> List[Dict[str, Any]]:
        """
        Lists all namespaces in the Pinecone database.

        Returns:
            List[Dict[str, Any]]: List of multiple namespace names, dimensions, vector counts.

        Raises:
            Exception: If listing namespaces fails.

        Example:
            >>> namespaces = client.get_namespaces()
        """
        try:
            pc = self.pinecone
            index = pc.Index(self.index_name)
            namespaces_data = index.describe_index_stats()
            namespaces_detail = [
                {
                    "namespaces": namespace_name,
                    "dimensions": namespaces_data["dimension"],
                    "vector_count": namespace_details["vector_count"],
                }
                for namespace_name, namespace_details in namespaces_data[
                    "namespaces"
                ].items()
            ]

            return namespaces_detail
        except PineconeApiException as e:
            _logger.error("Pinecone API error while listing namespaces: %s", e)
            raise VectorOperationError(
                f"Pinecone API error while listing namespaces: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to list namespaces: %s", e)

    def delete_namespace(self, namespace: str) -> None:
        """
        Deletes a namespace from the Pinecone database.

        Args:
            namespace (str): The namespace to delete.

        Returns:
            None

        Raises:
            Exception: If deleting the namespace fails.

        Example:
            >>> client.delete_namespace(namespace="example-ns")
        """
        try:
            if not namespace:
                raise VectorOperationError("Namespace cannot be empty")
            namespace_details = self.get_namespaces_detail()
            if not any(ns["namespaces"] == namespace for ns in namespace_details):
                raise VectorOperationError(f"Namespace '{namespace}' does not exist")
            index = self.pinecone.Index(self.index_name)
            index.delete(delete_all=True, namespace=namespace)
            _logger.info(f"Successfully deleted namespace '{namespace}'")
        except PineconeApiException as e:
            _logger.error("Pinecone API error while deleting namespace: %s", e)
            raise VectorOperationError(
                f"Pinecone API error while deleting namespace: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to delete namespace: %s", e)
            raise VectorOperationError(f"Failed to delete namespace: {str(e)}")

    def delete_all_namespaces(self) -> None:
        """
        Deletes all namespaces from the Pinecone database.

        Returns:
            None

        Raises:
            Exception: If deleting namespaces fails.

        Example:
            >>> client.delete_all_namespaces()
        """
        try:
            index = self.pinecone.Index(self.index_name)
            namespaces_details = self.get_namespaces_detail()
            for namespace in [ns["namespaces"] for ns in namespaces_details]:
                index.delete(delete_all=True, namespace=namespace)
            _logger.info("Successfully deleted all namespaces")
        except PineconeApiException as e:
            _logger.error("Pinecone API error while deleting namespaces: %s", e)
            raise VectorOperationError(
                f"Pinecone API error while deleting namespaces: {str(e)}"
            )
        except Exception as e:
            _logger.info("Failed to delete namespaces: %s", e)
            raise VectorOperationError(f"Failed to delete namespaces: {str(e)}")


class CustomJSONLoader:
    """
    Custom JSON loader class that does not depend on `jq` for loading JSON files into LangChain Documents.
    """

    def __init__(self, file_path: str):
        """
        Initialize the CustomJSONLoader with the file path of the JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Raises:
            ImportError: If the json module is not found.
        """
        try:
            import json
        except ImportError:
            _logger.error(
                "json is not installed. Please install it with `pip install json`."
            )
            raise ImportError(
                "json is not installed. Please install it with `pip install json`."
            )

        self.file_path = file_path
        self.json = json

    def load_json(self) -> List[Document]:
        """
        Load JSON file into a list of LangChain Document objects.

        Returns:
            List[Document]: A list containing a single Document with the JSON content.

        Raises:
            FileNotFoundError: If the JSON file is not found at the specified path.
            json.JSONDecodeError: If the JSON file cannot be decoded.
        """
        try:
            with open(self.file_path, "r") as file:
                data = self.json.load(file)
                _logger.debug(f"Successfully loaded JSON file from {self.file_path}")
        except FileNotFoundError as e:
            _logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}") from e
        except json.JSONDecodeError as e:
            _logger.error(f"Error decoding JSON from file: {self.file_path}")
            raise json.JSONDecodeError(
                f"Error decoding JSON from file: {self.file_path}"
            ) from e

        metadata = {"source": str(self.file_path)}
        return [Document(page_content=data, metadata=metadata)]


class PineConeDataLoader:
    """
    A loader that handles the loading of multiple data formats into a list of Langchain Document objects.
    Supports CSV, text, markdown, PDF, JSON, Excel, and DOCX formats and can embed the loaded data into a vector store.

    Class Constants:
        HUGGINGFACE_EMBEDDING_MODEL (str): Default HuggingFace model used for embedding.
        EMBEDDING_LOG_MSG (str): Log message for loading HuggingFace embeddings.
        HUGGINGFACE_CACHED_MODEL (or the cached HuggingFace model.

    Args:
        files (str | List[str]): Path to a file or a list of file paths.
        splitter (Any, optional): A text splitter for splitting documents into chunks.
        metadata (Dict[str, str], optional): Optional metadata for the documents.

    Example:
        >>> loader = PineConeDataLoader(files=["data.csv", "document.pdf"])
        >>> loader.load_data_into_vector_store()
    """

    def __init__(
        self,
        files: str | List[str] = None,
        splitter: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.files = files
        self.splitter = splitter
        self.metadata = metadata

    def load_data(self, format: str, data_path: str) -> List[Document]:
        """
        Loads data from the specified path based on the provided format.

        Args:
            format (str): The format of the file (csv, pdf, text, md, json, xlsx, docx).
            data_path (str): The file path to load data from.

        Returns:
            List[Document]: A list of Document objects.

        Raises:
            ImportError: If the necessary module for the file format is not installed.
            ValueError: If the provided file format is unsupported.
            Exception: For any other errors during data loading.

        Example:
            >>> loader = MultiDataLoader(files="data.csv")
            >>> documents = loader.load_data(format="csv", data_path="data.csv")
        """
        data = []
        try:
            # Load data based on the format
            if format == "csv":
                from langchain_community.document_loaders.csv_loader import CSVLoader

                loader = CSVLoader(file_path=data_path)
            elif format in ["txt", "md"]:
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(file_path=data_path)
            elif format == "pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(file_path=data_path)
            elif format == "json":
                loader = CustomJSONLoader(
                    file_path=data_path
                )  # Assuming CustomJSONLoader is defined elsewhere
            elif format == "xlsx":
                from langchain_community.document_loaders import UnstructuredExcelLoader

                loader = UnstructuredExcelLoader(
                    file_path=data_path, sheet_name=None, use_pandas=True
                )
            elif format == "docx":
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(file_path=data_path)
            elif format == "pptx":
                from langchain_community.document_loaders import (
                    UnstructuredPowerPointLoader,
                )

                loader = UnstructuredPowerPointLoader(file_path=data_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Load data depending on format
            data = loader.load_json() if format == "json" else loader.load()

        except ImportError as e:
            _logger.error(
                f"Import error: {e}. Ensure the required module for {format} format is installed."
            )
            raise ImportError(
                f"To use {format} format, please install langchain-community."
            ) from e
        except Exception as e:
            _logger.error(f"Error loading {format.upper()} data: {e}")
            raise e

        return data

    @staticmethod
    def _get_file_extensions(filename: str) -> str:
        """
        Extract the file extension from a filename.

        Args:
            filename (str): The filename to extract the extension from.

        Returns:
            str: The file extension.

        Example:
            >>> extension = MultiDataLoader._get_file_extensions("data.csv")
            >>> print(extension)  # Output: "csv"
        """
        return filename.split(".")[-1]

    def load_data_into_vector_store(
        self, index_name: str, namespace: Optional[str] = None
    ) -> None:
        """
        Loads data from the provided files, splits them into chunks (if a splitter is provided),
        generates embeddings, and adds them to the vector store.

        Default Splitter: RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=0.
        Default Embedding: HuggingFace Embeddings. (MODEL - sentence-transformers/all-mpnet-base-v2)

        Args:
            collection_name (str): Name of the collection to store the data in.

        Raises:
            ValueError: If there is an error with loading or embedding the data.
            Exception: For any general errors.

        Example:
            >>> loader = ChromaDataLoader(files=["data.csv", "document.pdf"])
            >>> loader.load_data_into_vector_store()
        """
        # Ensure `files` is a list of file paths
        files = [self.files] if isinstance(self.files, str) else self.files

        try:
            # Load documents from files
            documents = [
                doc
                for file in files
                for doc in self.load_data(
                    format=self._get_file_extensions(file), data_path=file
                )
            ]

            # Split documents into chunks if a splitter is provided, otherwise use default
            if self.splitter:
                _logger.info("Using provided text splitter.")
                docs = self.splitter.split_documents(documents)
            else:
                _logger.info(
                    "No splitter provided, using RecursiveCharacterTextSplitter with default settings."
                )
                docs = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0
                ).split_documents(documents)

            # Initialize ChromaClient for adding documents to the vector store
            pineconeclient = PineConeClient(
                api_key=os.getenv("PINECONE_API_KEY", "your-pinecone-api-key"),
                index_name=index_name,
            )

            # Add documents to the vector store using ChromaClient

            pineconeclient.add_data_to_vector_database(
                documents=docs,
                uuids=[str(uuid4()) for _ in range(len(docs))],
                namespace=namespace,
            )
            _logger.info("Successfully loaded data into vector store.")

        except Exception as e:
            _logger.error(f"Failed to load data into vector store: {e}")
            raise ValueError(f"Failed to load data into vector store: {e}")

    def _validate_batch(self, file_batch: List[Dict[str, Any]]) -> bool:
        """
        Validates a batch of file entries. Each entry in the batch should be a dictionary
        with the required keys and types to ensure proper data structure.

        Validation Rules:
        -----------------
        - Each file entry must be a dictionary.
        - Each file must contain a "file_path" key.
        - "file_path" must be a list of dictionaries, each containing a single path key with a nested dictionary.
        - The nested dictionary within "file_path" must contain a "metadata" key, which should itself be a dictionary.
        - Each file must contain a "collection_name" key, which must be a string.

        Args:
        ------
            file_batch (List[Dict[str, Any]]): List of file entries, where each entry is a dictionary
                                            with required keys and types.

        Returns:
        --------
            bool: True if all entries in the batch pass validation, otherwise raises a ValueError.

        Raises:
        -------
            ValueError: If any file in the batch does not meet the validation criteria.

        Example:
        --------
            >>> loader = ChromaDataLoader()
            >>> loader._validate_batch(file_batch=[{"file_path": [{"path": {"metadata": {}}}], "collection_name": "my_collection"}])

        Batch Format:
        -------------
        ```json
        [
            {
                "file_path": [
                    {"/path/to/file/1": {"metadata": {"key": "value"}}},
                    {"/path/to/file/2": {"metadata": {"key": "value"}}},
                    {"/path/to/file/3": {"metadata": {"key": "value"}}},
                ],
                "collection_name": "collection_name1",
                "collection_metadata": {"key": "value"},
            },
            {
                "file_path": [
                    {"/path/to/file/4": {"metadata": {"key": "value"}}},
                    {"/path/to/file/5": {"metadata": {"key": "value"}}},
                    {"/path/to/file/6": {"metadata": {"key": "value"}}},
                ],
                "collection_name": "collection_name2",
                "collection_metadata": {"key": "value"},
            },
            {
                "file_path": [
                    {"/path/to/file/7": {"metadata": {"key": "value"}}},
                    {"/path/to/file/8": {"metadata": {"key": "value"}}},
                    {"/path/to/file/9": {"metadata": {"key": "value"}}},
                ],
                "collection_name": "collection_name3",
                "collection_metadata": {"key": "value"},
            },
        ]
        ```
        """
        for idx, file in enumerate(file_batch):
            self._validate_file_entry(file, idx)
        _logger.info("Batch validation passed successfully.")
        return True

    def _validate_file_entry(self, file: Dict[str, Any], idx: int) -> None:
        """Validates the main file entry structure."""
        if not isinstance(file, dict):
            _logger.warning(f"File at index {idx} is not a dictionary.")
            raise ValueError(f"File at index {idx} is not a dictionary.")

        self._validate_file_path(file.get("file_path"), idx)

    def _validate_file_path(self, file_path: Any, file_idx: int) -> None:
        """Validates the 'file_path' structure."""
        if file_path is None:
            _logger.warning(f"'file_path' key missing in file at index {file_idx}.")
            raise ValueError(f"'file_path' key missing in file at index {file_idx}.")

        if not isinstance(file_path, list):
            _logger.warning(f"'file_path' is not a list in file at index {file_idx}.")
            raise ValueError(f"'file_path' must be a list in file at index {file_idx}.")

        for path_idx, path_dict in enumerate(file_path):
            self._validate_path_dict(path_dict, file_idx, path_idx)

    def _validate_path_dict(
        self, path_dict: Dict[str, Any], file_idx: int, path_idx: int
    ) -> None:
        """Validates individual path dictionaries within 'file_path'."""
        if len(path_dict) != 1:
            _logger.warning(
                f"Each entry in 'file_path' must contain a single path key in file at index {file_idx}, path entry {path_idx}."
            )
            raise ValueError(
                f"Each entry in 'file_path' must contain a single path key in file at index {file_idx}, path entry {path_idx}."
            )

        path_key, metadata_dict = next(iter(path_dict.items()))
        if not isinstance(metadata_dict, dict):
            _logger.warning(
                f"The value of the path '{path_key}' must be a dictionary in file at index {file_idx}, path entry {path_idx}."
            )
            raise ValueError(
                f"The value of the path '{path_key}' must be a dictionary in file at index {file_idx}, path entry {path_idx}."
            )

        self._validate_metadata(metadata_dict.get("metadata"), file_idx, path_idx)

    def _validate_metadata(self, metadata: Any, file_idx: int, path_idx: int) -> None:
        """Validates the 'metadata' dictionary within each path entry."""
        if metadata is None:
            _logger.warning(
                f"'metadata' key missing in file at index {file_idx}, path entry {path_idx}."
            )
            raise ValueError(
                f"'metadata' key missing in file at index {file_idx}, path entry {path_idx}."
            )

        if not isinstance(metadata, dict):
            _logger.warning(
                f"'metadata' must be a dictionary in file at index {file_idx}, path entry {path_idx}."
            )
            raise ValueError(
                f"'metadata' must be a dictionary in file at index {file_idx}, path entry {path_idx}."
            )

    def create_batches(
        self, data: List[Dict[str, Any]], batch_size: int
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Creates batches from the 'file_path' entries within each dictionary in the data list.

        Args:
            data (List[Dict[str, Any]]): The input data list, containing dictionaries with 'file_path' and 'collection_name'.
            batch_size (int): The number of file path entries in each batch.

        Yields:
            Generator[List[Dict[str, Any]]]: A generator yielding each batch of 'file_path' entries as a list.

        Raises:
            ValueError: If `batch_size` is less than 1.
            KeyError: If 'file_path' key is missing from any dictionary in `data`.
        """
        if batch_size < 1:
            _logger.error("Invalid batch_size: must be greater than 0.")
            raise ValueError("batch_size must be greater than 0.")

        if not isinstance(data, list):
            _logger.error("Data must be a list.")
            raise TypeError("Data must be a list.")

        # Generate batches of file entries
        for idx in range(0, len(data), batch_size):
            yield data[idx : min(idx + batch_size, len(data))]
