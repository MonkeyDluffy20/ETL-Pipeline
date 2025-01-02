"""
Vector Data Management and Synchronization Module

This module provides functionality to synchronize data between a SQL database and a Pinecone vector store.
It handles the creation, updating, and deletion of vector embeddings and their associated metadata,
ensuring consistency between the traditional database and vector store.

The module manages two main data stores:
1. SQL Database: Stores method metadata and tracking information
2. Pinecone Vector Store: Stores vector embeddings for semantic search capabilities

Dependencies:
    - langchain_core.documents: For document representation
    - utils.pinecone_wrapper: Custom wrapper for Pinecone operations
    - utils.db_manager: Custom SQL database manager
    - os: For environment variable access
    - time: For timestamp generation
    - uuid: For unique identifier generation
    - logging: For operation logging

Environment Variables Required:
    - PINECONE_API_KEY: API key for Pinecone vector database access

Example Usage:
    >>> method_ids = ["123", "456"]
    >>> method_data = [
    ...     {
    ...         "metadata": {
    ...             "method_name": "Test Method",
    ...             "segment": "Segment A",
    ...             "category": "Category 1",
    ...             "created_by": "User1",
    ...             "updated_by": "User2",
    ...             "method_id": "123"
    ...         },
    ...         "content": "This is test content for vector embedding."
    ...     },
    ...     # Additional method data...
    ... ]
    >>> update_data(method_ids, method_data)

Notes:
    - The module automatically handles both new insertions and updates
    - Vector deletions and additions are performed in bulk for efficiency
    - All operations are logged for debugging and monitoring
    - Timestamps are automatically generated and managed
"""

from langchain_core.documents import Document
from utils.pinecone_wrapper import PineConeClient
from utils.db_manager import SQLDatabase
import os
import time
from uuid import uuid4
from typing import List, Dict, Any
import logging

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def update_data(ids: List[str], data: List[Dict[str, Any]]) -> None:
    """
    Update or insert vector data and metadata for given method IDs in both SQL database and Pinecone vector store.

    This function performs a synchronization operation between a SQL database and Pinecone vector store:
    1. For each method ID, checks if it exists in the SQL database
    2. If it exists, updates the record with new data and generates a new UUID
    3. If it doesn't exist, creates a new record with a new UUID
    4. Deletes old vectors from Pinecone in bulk (for updated records)
    5. Adds new vectors to Pinecone in bulk

    The function handles both the SQL database operations and vector store operations atomically,
    ensuring consistency between both data stores.

    Args:
        ids (List[str]): List of method IDs to process. Each ID should correspond to a method
            that needs to be created or updated.
        data (List[Dict[str, Any]]): List of dictionaries containing method data and metadata.
            Each dictionary should have the following structure:
            {
                'metadata': {
                    'method_name': str,
                    'segment': str,
                    'category': str,
                    'created_by': str,
                    'updated_by': str,
                    'method_id': str
                },
                'content': str  # The actual content to be vectorized
            }

    Raises:
        ValueError: If PINECONE_API_KEY environment variable is not set.
        Exception: If there is any error during the database operations or vector processing.

    Notes:
        - The function generates new UUIDs for both new and updated records
        - Timestamps are automatically generated for tracking purposes
        - All operations are logged at DEBUG level
        - The function performs bulk operations on Pinecone for better performance
    """
    try:
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set as an environment variable.")

        pinecone_client = PineConeClient(
            api_key=api_key,
            index_name="glassco-index",
            dimension=1536,
            metric="dotproduct",
            embeddings="openai",
            deletion_protection="disabled",
        )

        # Initialize database connection
        db_manager = SQLDatabase()
        updated_documents = []  # For Pinecone
        existing_uuids_to_delete = []  # Track UUIDs to delete from Pinecone

        for method_id, method_data in zip(ids, data):
            # Check if method already exists in metadata table
            existing_query = """
            SELECT * FROM [dbo].[methodMetadata]
            WHERE method_id = ?
            """
            existing_metadata = db_manager.return_results(existing_query, (method_id,))

            current_time = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )  # Format current time for SQL Server
            new_uuid = str(uuid4())

            if existing_metadata:
                # If exists, collect old UUID for deletion
                old_uuid = existing_metadata[0]["uuid"]
                existing_uuids_to_delete.append(old_uuid)

                # Update metadata in SQL
                update_query = """
                UPDATE [dbo].[methodMetadata]
                SET uuid = ?,
                    methodName = ?,
                    segment = ?,
                    category = ?,
                    description = ?,
                    createdBy = ?,
                    updatedBy = ?,
                    vectorUpdatedAt = ?
                WHERE method_id = ?
                """
                update_params = (
                    new_uuid,
                    method_data["metadata"]["method_name"],
                    method_data["metadata"]["segment"],
                    method_data["metadata"]["category"],
                    method_data["content"],
                    method_data["metadata"]["created_by"],
                    method_data["metadata"]["updated_by"],
                    current_time,
                    method_id,
                )
                db_manager.execute_query(update_query, update_params)

            else:
                # Insert new metadata record
                insert_query = """
                INSERT INTO [dbo].[methodMetadata]
                (method_id, uuid, methodName, segment, category, description,
                createdBy, updatedBy, vectorCreatedAt, vectorUpdatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                insert_params = (
                    method_id,
                    new_uuid,
                    method_data["metadata"]["method_name"],
                    method_data["metadata"]["segment"],
                    method_data["metadata"]["category"],
                    method_data["content"],
                    method_data["metadata"]["created_by"],
                    method_data["metadata"]["updated_by"],
                    current_time,
                    None,
                )
                db_manager.execute_query(insert_query, insert_params)

            # Update document metadata
            method_data["metadata"].update(
                {
                    "uuid": new_uuid,
                    "vectorCreatedAt": current_time,
                    "vectorUpdatedAt": None if not existing_metadata else current_time,
                }
            )

            # Add to updated documents list
            updated_documents.append(
                Document(
                    page_content=method_data["content"],
                    metadata={
                        key: (value if value is not None else "")
                        for key, value in method_data["metadata"].items()
                    },
                )
            )

        # Delete existing vectors from Pinecone in bulk
        if existing_uuids_to_delete:
            _logger.info(
                f"Deleting {len(existing_uuids_to_delete)} existing vectors from Pinecone"
            )
            pinecone_client.delete_data_by_id(ids=existing_uuids_to_delete)

        # Add all updated documents to Pinecone
        if updated_documents:
            _logger.info(f"Adding {len(updated_documents)} vectors to Pinecone")
            pinecone_client.add_data_to_vector_database(
                documents=updated_documents,
                uuids=[doc.metadata["uuid"] for doc in updated_documents],
            )

        _logger.info(f"Successfully processed {len(ids)} methods")

    except Exception as e:
        _logger.error(f"Error during data processing pipeline: {e}")
        raise
