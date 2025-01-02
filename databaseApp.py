"""
Azure Function Service Bus Queue Handler for Vector Data Management

This Azure Function handles messages from a Service Bus queue to manage vector data in a Pinecone database.
It processes incoming method IDs, extracts corresponding data, and synchronizes it with both SQL and 
vector databases.

Flow:
1. Receives message from Azure Service Bus queue containing comma-separated method IDs
2. Validates the format of received IDs
3. Extracts and filters data for the requested methods
4. Updates both SQL database and Pinecone vector store with the processed data

Dependencies:
    - azure.functions: For Azure Function and Service Bus integration
    - utils.db_script: Custom data processing utility
    - utils.upsert_database: Database update operations
    - logging: For operation logging and monitoring
    - os, sys: For path management and environment settings

Environment Setup:
    - Requires appropriate Azure Function configuration
    - Needs access to Service Bus queue
    - Requires database connection settings
    - Requires Pinecone API credentials

Example Message Format:
    "123,456,789"  # Comma-separated method IDs

Notes:
    - The function processes one message at a time
    - All method IDs must be numeric
    - Logs detailed information at various stages for monitoring
    - Includes error handling and validation
"""

import os
import sys

# Add necessary paths to system path for importing local modules
# This ensures access to utility modules in different directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ServiceBusQueueTrigger1")
    ),
)

import logging
from azure.functions import ServiceBusMessage
from utils.db_script import DataProcessor
from utils.upsert_database import update_data

# Configure logging with timestamp and level information
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(msg: ServiceBusMessage) -> None:
    """
    Process Service Bus queue messages and update vector database accordingly.

    This function is triggered by messages in an Azure Service Bus queue. It processes
    the message content (comma-separated method IDs), validates the IDs, retrieves
    corresponding data, and updates both SQL and vector databases with the processed
    information.

    The function follows these steps:
    1. Decodes and parses the queue message
    2. Validates all method IDs are numeric
    3. Extracts and filters data for the specified methods
    4. Updates databases with the processed information
    5. Logs the operation status and any errors

    Args:
        msg (ServiceBusMessage): Azure Service Bus message object containing
            comma-separated method IDs in its body.

    Raises:
        ValueError: When message format is invalid or when no matching data is found
            for the provided method IDs.
        Exception: For other processing errors (e.g., database connection issues,
            data processing failures).

    Notes:
        - All operations are logged for monitoring and debugging
        - Failed operations include detailed error information
        - Optional debug file output is available for troubleshooting
    """
    try:
        # Extract and decode message content from Service Bus queue
        # ServiceBusMessage body needs UTF-8 decoding
        queue_msg = msg.get_body().decode("utf-8")
        _logger.info(
            "Python ServiceBus queue trigger processed message: %s",
            queue_msg,
        )

        # Parse message into list of IDs and remove any whitespace
        ids_list = queue_msg.split(",")
        _logger.debug(f"Parsed IDs list: {ids_list}")

        # Validate each ID is numeric to prevent processing invalid data
        for id_value in ids_list:
            id_value = id_value.strip()
            if not id_value.isdigit():
                error_msg = f"Invalid ID format: '{id_value}'. All IDs must be numeric."
                _logger.error(error_msg)
                raise ValueError(error_msg)

        # Initialize data processor and extract all available data
        data_processor = DataProcessor()
        extracted_data = data_processor.extract_and_format_data()
        _logger.debug(f"Total extracted data records: {len(extracted_data)}")

        # Filter the extracted data to include only requested method IDs
        filtered_data = [
            data
            for data in extracted_data
            if str(data["metadata"]["method_id"]) in ids_list
        ]
        _logger.debug(f"Filtered data records: {len(filtered_data)}")
        _logger.debug(
            f"Filtered data method IDs: {[data['metadata']['method_id'] for data in filtered_data]}"
        )

        # Verify that matching data was found for the requested IDs
        if not filtered_data:
            _logger.error(f"No matching data found for method IDs: {ids_list}")
            raise ValueError(f"No data found for requested method IDs: {ids_list}")

        # Save individual text files for debugging purposes (optional)
        data_processor.save_to_individual_txt_files(filtered_data)

        # Update databases with the processed data
        # This updates both SQL and vector databases
        update_data(ids_list, filtered_data)

        _logger.info("Successfully completed processing all methods")

    except Exception as e:
        # Log detailed error information for troubleshooting
        _logger.error(f"Error processing message: {str(e)}", exc_info=True)
        _logger.info(f"Failed IDs: {ids_list}")
        _logger.info(f"Failure message: {str(e)}")
        raise  # Re-raise the exception for Azure Function runtime handling
