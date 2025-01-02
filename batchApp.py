# ETL Pipeline with Azure Service Bus Trigger

"""
This module implements an ETL pipeline triggered by messages from an Azure Service Bus queue. 
The pipeline extracts file metadata based on provided IDs, processes the metadata, 
and stores the results in a vector store and SQL Server.

Modules:
    - SQLDatabase: Manages database connections and executes SQL queries.
    - BatchProcessor: Handles batch processing of metadata for efficient data handling.

Workflow:
    1. Message received from Azure Service Bus queue.
    2. Metadata queried from SQL Server based on provided file IDs.
    3. Metadata processed in batches and stored in the appropriate vector store and database.

Dependencies:
    - azure.functions: For interacting with Azure Service Bus queues.
    - utils.db_manager: Provides database management functionalities.
    - utils.batch_processor: Handles batch data processing.

"""

# Add multiple paths to ensure imports work
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "BBGDServiceBusQueueTrigger")
    ),
)

# Import Dependencies
import logging
from azure.functions import ServiceBusMessage
from utils.db_manager import SQLDatabase
from utils.batch_processor import BatchProcessor

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(msg: ServiceBusMessage):
    """
    Azure Service Bus Trigger Function

    This function is triggered when a message is received from the Azure Service Bus queue.
    It extracts file IDs from the message, queries metadata from SQL Server, and processes
    the metadata using the BatchProcessor.

    Args:
        msg (ServiceBusMessage): The message received from the Azure Service Bus queue,
                                 containing a comma-separated list of file IDs.

    Workflow:
        1. Parse the message to extract file IDs.
        2. Query metadata from SQL Server using the extracted IDs.
        3. Process the metadata using the BatchProcessor.
        4. Log the processing results and any failures.

    """
    # Get the message from the Service Bus queue and log it
    queue_msg = msg.get_body().decode("utf-8")
    _logger.info(
        "Python ServiceBus queue trigger processed message: %s",
        queue_msg,
    )

    # Process the message (ETL)
    # Extract the data from SQL Server and Azure Blob Storage
    db_manager = SQLDatabase(
        server="ASUS",
        database="BBGD",
        local=True,
    )

    # Split the message into a list of IDs
    ids_list = queue_msg.split(",")
    # Format each ID with single quotes and join them with commas
    formatted_ids = ", ".join(f"'{id.strip()}'" for id in ids_list)

    # Construct the final SQL query
    query = f"SELECT * FROM file_metadata WHERE file_id IN ({formatted_ids});"
    _logger.info(f"Query: {query}")

    metadata = db_manager.return_results(query=query)

    _logger.info(f"Metadata: {metadata}")

    processor = BatchProcessor(data=metadata)
    processor.process_batches_parallel()

    _logger.info(f"Failed Ids {processor.failed_file_ids}")
    _logger.info(f"Failure message: {processor.failed_message}")
