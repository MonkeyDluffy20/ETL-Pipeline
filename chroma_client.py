"""
This module provides functionality to interact with a Chroma vector database 
service for storing and querying document embeddings. It defines the following 
classes and utilities:

Classes:
    - `ChromaClient`: Handles communication with the Chroma service, 
        providing methods to create collections, add documents, and query the vector store.
    - `CustomErrorMessage`: A custom exception class for handling credential-related 
        errors.
    - `CustomJSONLoader`: A utility class for loading JSON files into LangChain 
        `Document` objects, designed to work without external dependencies like `jq`.
    - `ChromaDataLoader`: A versatile loader that supports multiple file formats 
        (CSV, PDF, JSON, DOCX, Excel, etc.) and embeds the loaded data into a Chroma 
        vector store.

Features:
    - Multi-tenant and multi-database support for Chroma.
    - Integration with the HuggingFace embedding model for document embeddings.
    - Support for multiple document formats, enabling the extraction and storage of 
        data from CSV, PDF, JSON, DOCX, and other files.
    - Efficient text splitting through LangChain's `RecursiveCharacterTextSplitter` 
        for chunking large documents.

Usage Example:
    ```python
    # Initialize ChromaClient
    client = ChromaClient(host="localhost", port=8000)

    # Add documents to a vector store
    documents = [Document(page_content="sample text", metadata={})]
    uuids = ["uuid1"]
    client.add_documents_to_vector_store(
        collection_name="my_collection", 
        documents=documents, 
        embedding_function=my_embedding_function, 
        uuids=uuids
    )

    # Query the vector store
    similar_docs = client.query_vector_store(query="search query", k=3)
    ```

Logging:
    The module integrates a logger for tracking errors, warnings, and key events 
    during execution. Logs are created using the `create_logger` function.

Exceptions:
    Custom exceptions are used to raise meaningful error messages when issues occur 
    (e.g., `CustomErrorMessage`).
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import pysqlite3 to replace sqlite3
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Import Dependencies
import json
import logging
from uuid import uuid4
from datetime import datetime

from chromadb import HttpClient, PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List, Any, Dict, Optional, Sequence

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomErrorMessage(Exception):
    """Exception raised when credentials are not found."""

    def __init__(self, message: str) -> None:
        """Initializes the CustomErrorMessage with the given message.

        Args:
            message (str, optional): The error message.
        """
        super().__init__(message)


class ChromaClient:
    """
    ChromaClient handles the communication with a Chroma vector database service,
    allowing operations like creating collections, vector stores, and adding documents.

    Args:
        host (str): The hostname or IP address of the Chroma server.
        port (int): The port number to connect to Chroma.
        ssl (bool): Whether to use SSL for the connection.
        headers (Dict[str, str], optional): HTTP headers for the connection.
        settings (Settings, optional): Configuration settings for the connection.
        tenant (str): Tenant identifier for multi-tenant support. Defaults to DEFAULT_TENANT.
        database (str): Database identifier for multi-database support. Defaults to DEFAULT_DATABASE.

    Example:
        >>> client = ChromaClient(host="127.0.0.1", port=8000, ssl=False)
        >>> collection = client._create_collection("my_collection")
    """

    COLLECTION_EMPTY_ERROR_MSG = "Collection name cannot be empty."

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        use_persistent: bool = False,
        headers: Optional[Dict[str, str]] = None,
        settings: Optional[Settings] = None,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ) -> None:
        """Initialize ChromaClient and create an HTTP client instance.

        Raises:
            Exception: If the HTTP client fails to initialize.

        Example:
            >>> client = ChromaClient(host="localhost", port=8000)
        """
        self.host = host
        self.port = port
        self.ssl = ssl
        self.headers = headers
        self.settings = settings
        self.tenant = tenant
        self.database = database
        self.use_persistent = use_persistent

        if self.use_persistent:
            _logger.info("Using persistent Client")
        else:
            _logger.info(
                f"Initializing ChromaClient for {self.host}:{self.port} with SSL={self.ssl}"
            )

        # Try to initialize the client, log any issues
        try:
            if not use_persistent:
                self.client = HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=self.ssl,
                    headers=self.headers,
                    settings=self.settings,
                    tenant=self.tenant,
                    database=self.database,
                )
                _logger.info(
                    f"ChromaClient initialized successfully for tenant '{self.tenant}' and database '{self.database}'."
                )
            else:
                self.client = PersistentClient(
                    path="./chroma",
                    settings=self.settings,
                    tenant=self.tenant,
                    database=self.database,
                )
        except Exception as e:
            _logger.error(f"Failed to initialize ChromaClient: {e}")
            raise CustomErrorMessage(f"Failed to initialize ChromaClient: {e}")

    def _create_collection(
        self, collection_name: str, type: str = "Document"
    ) -> Collection:
        """
        Create or retrieve an existing collection by name in Chroma.

        Args:
            collection_name (str): Name of the collection to create or retrieve.
            type (str, optional): Type of the collection. Defaults to "Document".

        Returns:
            Collection: The Chroma collection object.

        Raises:
            ValueError: If the collection name is empty.
            Exception: If there is an issue retrieving or creating the collection.

        Example:
            >>> client = ChromaClient()
            >>> collection = client._create_collection("my_collection")
        """
        if not collection_name:
            _logger.error(ChromaClient.COLLECTION_EMPTY_ERROR_MSG)
            raise ValueError(ChromaClient.COLLECTION_EMPTY_ERROR_MSG)

        try:
            _logger.info(f"Creating or retrieving collection '{collection_name}'")
            return self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": type,
                },
            )
        except Exception as e:
            _logger.error(f"Failed to create or retrieve collection: {e}")
            raise CustomErrorMessage(f"Failed to create or retrieve collection: {e}")

    def _create_vector_store(
        self, collection_name: str, embedding_function: Any
    ) -> Chroma:
        """
        Create a vector store (Chroma) for storing document embeddings.

        Args:
            collection_name (str): Name of the collection to store embeddings.
            embedding_function (Any): A function to generate embeddings from documents.

        Returns:
            Chroma: The vector store object.

        Raises:
            ValueError: If the collection name or embedding function is invalid.
            ImportError: If Chroma is not properly imported or initialized.
            Exception: For any other issues during vector store creation.

        Example:
            >>> client = ChromaClient()
            >>> vector_store = client._create_vector_store("my_collection", embedding_function=my_embedding_function)
        """
        if not collection_name or not embedding_function:
            _logger.error("Invalid collection name or embedding function.")
            raise ValueError("Collection name and embedding function must be provided.")

        try:
            _logger.info(f"Creating vector store for collection '{collection_name}'")
            return Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=embedding_function,
                create_collection_if_not_exists=False,
            )
        except ImportError as e:
            _logger.error(f"Chroma import failed: {e}")
            raise ImportError(f"Chroma import failed: {e}")
        except Exception as e:
            _logger.error(f"Failed to create vector store: {e}")
            raise CustomErrorMessage(f"Failed to create vector store: {e}")

    def add_documents_to_vector_store(
        self,
        collection_name: str,
        documents: List[Document],
        embedding_function: Any,
        uuids: List[str],
    ) -> None:
        """
        Add a list of documents to the vector store after generating their embeddings.

        Args:
            collection_name (str): Name of the collection to add documents to.
            documents (List[Document]): List of documents to store in the vector store.
            embedding_function (Any): Function to generate embeddings for documents.
            uuids (List[str]): Unique identifiers (UUIDs) for the documents.

        Raises:
            ValueError: If any of the required parameters are invalid.
            Exception: If document addition fails.

        Example:
            >>> client = ChromaClient()
            >>> documents = [Document("doc1", "content1"), Document("doc2", "content2")]
            >>> client.add_documents_to_vector_store("my_collection", documents, embedding_function=my_embedding_function, uuids=["id1", "id2"])
        """
        if not collection_name or not documents or not embedding_function or not uuids:
            _logger.error("Invalid parameters for adding documents to vector store.")
            raise ValueError(
                "Collection name, documents, embedding function, and UUIDs are required."
            )

        try:
            # Ensure the collection and vector store exist
            vector_store = self._create_vector_store(
                collection_name, embedding_function
            )

            _logger.info(
                f"Adding {len(documents)} documents to the vector store '{collection_name}'"
            )
            vector_store.add_documents(documents=documents, ids=uuids)
            _logger.info(
                f"Successfully added documents to vector store '{collection_name}'"
            )
        except Exception as e:
            _logger.error(f"Failed to add documents to vector store: {e}")
            raise CustomErrorMessage(f"Failed to add documents to vector store: {e}")

    def query_vector_store(
        self,
        query: str,
        collection_name: str,
        k: int = 3,
        filters: Dict[str, str] = None,
    ) -> List[str]:
        """
        Query the vector store for similar documents based on the provided query.

        Args:
            query (str): The query string to search for similar documents.
            collection_name (str): Name of the collection to query.
            k (int, optional): The number of similar documents to return. Defaults to 3.
            filters (Dict[str, str], optional): Filters to apply to the query. Defaults to None.

        Returns:
            List[str]: A list of document IDs similar to the query.

        Raises:
            ValueError: If the query is empty or invalid.
            Exception: If the query fails.

        Example:
            >>> client = ChromaClient()
            >>> similar_docs = client.query_vector_store(query="search query", k=3)
        """
        if not query:
            _logger.error("Query cannot be empty.")
            raise ValueError("Query cannot be empty.")

        try:
            _logger.info(f"Querying vector store for similar documents to '{query}'")
            # Ensure the collection and vector store exist
            vector_store = self._create_vector_store(
                collection_name=collection_name,
                embedding_function=ChromaDataLoader._create_embeddings(),
            )
            return vector_store.similarity_search(query=query, k=k, filter=filters)
        except Exception as e:
            _logger.error(f"Failed to query vector store: {e}")
            raise CustomErrorMessage(f"Failed to query vector store: {e}")

    def delete_collection(self, collection_name: str) -> None:
        """ """
        if not collection_name:
            _logger.error("collection name cannot be empty.")
            raise ValueError("collection name cannot be empty.")

        try:
            _logger.info(f"Deleting collection '{collection_name}'")
            # Ensure the collection and vector store exist
            vector_store = self._create_vector_store(
                collection_name=collection_name,
                embedding_function=ChromaDataLoader._create_embeddings(),
            )
            vector_store.delete_collection()
            _logger.info(f"Successfully deleted collection '{collection_name}'")
        except Exception as e:
            _logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise CustomErrorMessage(
                f"Failed to delete collection {collection_name}: {e}"
            )

    def list_all_collections(self) -> Sequence[Collection]:
        """
        Retrieve a list of all collections within the ChromaDB client.

        This function calls the ChromaDB client to list all existing collections and logs
        the action, including any errors encountered.

        Returns:
            Sequence[Collection]: A sequence of ChromaDB collections, where each collection
            contains its metadata and data records as configured within the database.

        Raises:
            Exception: If an error occurs during the listing of collections, it is logged
            and re-raised for handling in the calling code.
        """
        try:
            _logger.info("Attempting to list all collections in the ChromaDB client.")
            collections = self.client.list_collections()
            _logger.info(f"Successfully retrieved {len(collections)} collections.")
            return collections

        except Exception as e:
            _logger.error("Error while listing collections: %s", e)
            raise e


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


class ChromaDataLoader:
    """
    A loader that handles the loading of multiple data formats into a list of Langchain Document objects.
    Supports CSV, text, markdown, PDF, JSON, Excel, and DOCX formats and can embed the loaded data into a vector store.

    Class Constants:
        HUGGINGFACE_EMBEDDING_MODEL (str): Default HuggingFace model used for embedding.
        EMBEDDING_LOG_MSG (str): Log message for loading HuggingFace embeddings.
        HUGGINGFACE_CACHED_MODEL (str): Path for the cached HuggingFace model.

    Args:
        files (str | List[str]): Path to a file or a list of file paths.
        use_local_persistent_path (bool): Whether to use Local File system as a persistent Database Volume for a client or
                                            Use Http Client (Where a Chroma DB is hosted).
        splitter (Any, optional): A text splitter for splitting documents into chunks.
        embedding_model (Any, optional): Embedding model to be used. Defaults to HuggingFace.
        metadata (Dict[str, str], optional): Optional metadata for the documents.

    Example:
        >>> loader = MultiDataLoader(files=["data.csv", "document.pdf"], use_local_persistent_path=False, splitter=my_splitter)
        >>> loader.load_data_into_vector_store()
    """

    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_LOG_MSG = "Loading HuggingFace Embeddings as default Embedding."
    HUGGINGFACE_CACHED_MODEL = "./models"

    def __init__(
        self,
        files: str | List[str],
        use_local_persistent_path: bool = False,
        splitter: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.files = files
        self.use_local_persistent_path = use_local_persistent_path
        self.splitter = splitter
        self.metadata = metadata
        self.embedding_model = embedding_model

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
            elif format in ["xlsx", "xls"]:
                from langchain_community.document_loaders import UnstructuredExcelLoader

                loader = UnstructuredExcelLoader(
                    file_path=data_path,
                    sheet_name=None,
                    engine="xlrd",
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

    @staticmethod
    def _create_embeddings():
        """
        Create an embedding model using HuggingFace.

        Returns:
            HuggingFaceEmbeddings: HuggingFace embedding model.

        Example:
            >>> embeddings = MultiDataLoader._create_embeddings()
        """
        from langchain_huggingface import HuggingFaceEmbeddings

        _logger.info(ChromaDataLoader.EMBEDDING_LOG_MSG)
        return HuggingFaceEmbeddings(
            model_name=ChromaDataLoader.HUGGINGFACE_EMBEDDING_MODEL,
            cache_folder=ChromaDataLoader.HUGGINGFACE_CACHED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    def load_data_into_vector_store(self, collection_name: str) -> None:
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

            # Create embeddings if not provided
            if not self.embedding_model:
                self.embedding_model = self._create_embeddings()

            # Initialize ChromaClient for adding documents to the vector store
            chroma_client = ChromaClient(use_persistent=self.use_local_persistent_path)

            # Get or create collection for storing the data
            collection = chroma_client._create_collection(collection_name)
            if not collection:
                _logger.error(f"Failed to create collection: {collection_name}")
                raise CustomErrorMessage(
                    f"Failed to create collection: {collection_name}"
                )

            # Add documents to the vector store using ChromaClient

            chroma_client.add_documents_to_vector_store(
                collection_name=collection_name,
                documents=docs,
                embedding_function=self.embedding_model,
                uuids=[str(uuid4()) for _ in range(len(docs))],
            )
            _logger.info("Successfully loaded data into vector store.")

            ####
            _logger.info(f"Document count in the collection: {collection.count()}")

        except Exception as e:
            _logger.error(f"Failed to load data into vector store: {e}")
            raise CustomErrorMessage(f"Failed to load data into vector store: {e}")
