# Import Dependencies
import logging
import datetime
from azure.storage.blob import BlobServiceClient


class ETLDetailsLogger:
    """
    A class to log chat messages into Azure Blob Storage.

    Attributes:
        connection_string (str): Azure Blob Storage connection string.
        container_name (str): The name of the blob container to store chat logs.
    """

    def __init__(self, connection_string: str, container_name: str):
        """
        Initializes the ChatLogger with Azure Blob Storage connection details.

        Args:
            connection_string (str): Azure Blob Storage connection string.
            container_name (str): The name of the blob container.
        """
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_name = container_name

        # Create the container if it does not exist
        self._create_container_if_not_exists()

    def _create_container_if_not_exists(self):
        """Creates the container in Azure Blob Storage if it doesn't exist."""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            if not container_client.exists():
                container_client.create_container()
        except Exception as e:
            logging.error(f"Error creating container: {e}")
            raise

    def log_service_queue_execution(self, chat_id: str, message: str):
        """
        Logs a Service Bus Queue message execution time to Azure Blob Storage.

        Args:
            chat_id (str): Unique identifier for the chat session.
            message (str): The chat message to log.
        """
        try:
            # Generate a timestamped blob name
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            blob_name = f"{chat_id}/{timestamp}.txt"

            # Upload the chat message as a blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            blob_client.upload_blob(message, overwrite=True)

            logging.info(f"Chat logged successfully: {blob_name}")
        except Exception as e:
            logging.error(f"Error logging chat: {e}")
            raise
