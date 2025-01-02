"""
BatchProcessor Module

This module defines the BatchProcessor class, which is responsible for processing
file metadata in batches. It creates batches of files based on their category and
loads them into a vector store using the ChromaClient and ChromaDataLoader. The
processing is done in parallel using a thread pool to improve efficiency.

Classes:
    - BatchProcessor: Handles batch processing of file metadata.

Usage Example:
    >>> batch_processor = BatchProcessor(data, client)
    >>> batch_processor.process_batches_parallel()

Dependencies:
    - logging: For logging information and errors
    - dataclasses: For converting FileMetadata objects to dictionaries
    - collections: For creating defaultdict and iterators
    - typing: For type hints
    - concurrent.futures: For parallel processing
"""

# Add directories to the system path for custom module imports
import sys

sys.path.append("./")
sys.path.append("../")

# Import Dependencies
import os
import logging
from dataclasses import asdict
from collections import defaultdict
from typing import List, Dict, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.azure_utils import AzureBlobStorage

from typing import Set      

from utils.data_class import FileMetadata
from utils.chroma_client import ChromaClient, ChromaDataLoader

# Configure logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BatchProcessor:
    def __init__(self, data: List[Dict[str, str]], # client: ChromaClient 
                ) -> None:
        """
        Initialize the BatchProcessor with file metadata and a Chroma client.

        Args:
            data (List[Dict[str, str]]): A list of file metadata dictionaries.
            client (ChromaClient): An instance of the ChromaClient class to interact with the vector store.

        Example:
            >>> batch_processor = BatchProcessor(data, client)
        """
        try:
            self.data = [FileMetadata(**item) for item in data]
            # self.client = client
            self.failed_file_ids: Set[str] = set()
            self.failed_message = []
            _logger.info("BatchProcessor initialized with %d files", len(self.data))
        except Exception as e:
            _logger.error("Failed to initialize BatchProcessor: %s", str(e))
            raise

    def create_batches(self) -> Dict[str, Iterator[FileMetadata]]:
        """
        Create batches of files grouped by their category.

        Returns:
            Dict[str, Iterator[FileMetadata]]: A dictionary where keys are categories
                                                and values are iterators of FileMetadata objects.
        """
        try:
            batches = defaultdict(list)
            for file in self.data:
                batches[file.category].append(file)

            category_dict = {category: iter(files) for category, files in batches.items()}
            _logger.info("Created %d batches", len(category_dict))
            return category_dict
        except Exception as e:
            _logger.error("Error creating batches: %s", str(e))
            raise

    def process_batch(self, batch: Iterator[FileMetadata], category: str):
        """
        Process a single batch of files by loading them into the vector store.
        This method iterates over the files in the batch, loads them into the vector store,
        and logs any errors that occur during the process.

        Args:
            batch (Iterator[FileMetadata]): An iterator of FileMetadata objects.
            category (str): The category of the files in the batch.

        Returns:
            None
        """
        try:
            _logger.info("Starting to process batch for category: %s", category)
            for file in batch:
                try:
                    _logger.info("Processing file: %s", file.filename)
                    # Initialize Azure Blob Service client
                    blob_service_client = AzureBlobStorage.get_blob_service_client(
                        connection_str=os.getenv("BLOB_STORAGE_CONNECTION_STRING"),
                        blob_service_endpoint=os.getenv("BLOB_SERVICE_ENDPOINT")
                    )
                    
                    file_name = file.full_path.split("/")[-1]
                    _logger.info(f"File name for downloading file in tmp - {file_name}")
                    
                    
                    local_file_path = AzureBlobStorage.download_file_to_tmp_dir(
                        container_name=os.getenv("BLOB_CONTAINER_NAME"),
                        file_name=file_name,
                        blob_service_client=blob_service_client,
                    )
                    
                    loader = ChromaDataLoader(
                        files=[local_file_path],
                        use_local_persistent_path=False,
                        metadata=asdict(file),
                    )
                    collection_name = file.category.replace(".", "")
                    loader.load_data_into_vector_store(collection_name=collection_name)
                except Exception as e:
                    _logger.info("Error processing file %s: %s", file.file_id, str(e))
                    self.failed_file_ids.add(file.file_id)
                    self.failed_message.append(str(e))
            _logger.info("Completed processing batch for category: %s", category)
        except Exception as e:
            _logger.error("Error processing batch for category %s: %s", category, str(e))
            raise

    def process_batches_parallel(self):
        """
        Process all file batches in parallel using a thread pool.

        Returns:
            None
        """
        try:
            batches = self.create_batches()

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_batch, batch, category): category
                    for category, batch in batches.items()
                }

                for future in as_completed(futures):
                    category = futures[future]
                    try:
                        future.result()
                        _logger.info("Finished processing category: %s", category)
                    except Exception as e:
                        _logger.error("Error in processing category %s: %s", category, str(e))
        except Exception as e:
            _logger.error("Error in parallel batch processing: %s", str(e))
            raise
