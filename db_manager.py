"""
SQLDatabase: A class to interact with an SQL database. It runs SQL queries against the database
    and returns the results in a structured format.
    
TODO:
1. Add support for multiple database connections.
2. Add support for different types of SQL databases (e.g., MySQL, PostgreSQL).
3. Add support for executing DDL queries (e.g., CREATE TABLE, ALTER TABLE).
4. Add support for logging and monitoring

Usage:
    db = SQLDatabase()
    results = db.return_results("SELECT * FROM <TABLE_NAME>")
    db.save_record({"column1": "value1", "column2": "value2"})
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing dependencies
import os
import pyodbc
import logging
from dotenv import load_dotenv
from pyodbc import Cursor

from typing import Any, Dict, List, Tuple

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Environment variables
load_dotenv(override=True)

# Define constants
DB_CONNECTING_ERROR_MESSAGE = "Error connecting to the database"


class SQLDatabase:
    """
    A class to interact with an SQL database.

    Attributes:
        connection_string (str): The connection string for the database.
    """

    def __init__(
        self,
        server: str = None,
        database: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        local: bool = True,
    ):
        """
        Initializes the SQLDatabase object with the connection parameters retrieved from arguments or environment variables.

        Args:
            server (str, optional): The SQL server address. Defaults to the value of the 'AZURE_SQL_SERVER' environment variable if not provided.
            database (str, optional): The name of the database. Defaults to the value of the 'AZURE_SQL_DATABASE' environment variable if not provided.
            user (str, optional): The username for database authentication. Defaults to the value of the 'AZURE_SQL_USER' environment variable if not provided.
            password (str, optional): The password for database authentication. Defaults to the value of the 'AZURE_SQL_PASSWORD' environment variable if not provided.
            port (int, optional): The port for database authentication. Defaults to the value of the 'AZURE_SQL_PORT' environment variable if not provided.
            local (bool): Whether to use local database. Defaults to True (For now, subjected to change based on deployment plans).
        Raises:
            ValueError: If any of the required connection parameters (server, database, user, password) are not provided either through arguments or environment variables.
        """
        self.server = server or os.getenv("AZURE_SQL_SERVER")
        self.database = database or os.getenv("AZURE_SQL_DATABASE")
        self.user = user or os.getenv("AZURE_SQL_USER")
        self.password = password or os.getenv("AZURE_SQL_PASSWORD")
        self.port = port or os.getenv("AZURE_SQL_PORT")
        self.local = local

        if not self.local and not all(
            [self.server, self.database, self.user, self.password, self.port]
        ):
            raise ValueError(
                "All connection parameters must be provided, either through arguments or environment variables."
            )

        self.connection_string = self._construct_connection_string()
        _logger.info("Connecting to %s", self.database)

    def _construct_connection_string(self):
        """
        Constructs the connection string based on the provided connection parameters.

        Returns:
            str: The constructed connection string.
        """
        if self.local:
            _logger.info("Connecting to local database")
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                "trusted_connection=yes;"
            )
        else:
            _logger.info("Connecting to Azure database")
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
                "TrustServerCertificate=no;"
                "Encrypt=yes;"
                "ConnectionTimeout=30;"
            )
        return connection_string

    def connect(self):
        """
        Connects to the database using the provided connection string.

        Returns:
            pyodbc.Connection or None: A connection object if successful, None otherwise.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            _logger.info("Connected to the database")
            return conn
        except Exception as e:
            _logger.error("Error connecting to the database: %s", e)
            return None

    def close(self, conn):
        """
        Closes the database connection.

        Args:
            conn (pyodbc.Connection): The connection object to be closed.
        """
        try:
            conn.close()
            _logger.info("Connection closed")
        except Exception as e:
            _logger.error("Error closing connection: %s", e)

    def execute_query(self, query: str) -> Tuple[List, Cursor]:
        """
        Executes an SQL query on the database.

        Args:
            query (str): The SQL query to be executed.
        """
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(query)
                records = cursor.fetchall()
                _logger.info("Query executed successfully")
                return records, cursor
            except Exception as e:
                _logger.error("Error executing query: %s", e)
                self.close(conn)
        else:
            _logger.error(DB_CONNECTING_ERROR_MESSAGE)

    def format_query_results(
        self, records: List[Tuple], cursor: Cursor
    ) -> List[Dict[str, Any]]:
        """
        Converts a query result into a list of dictionaries.

        Args:
            records (List[Tuple]): The query result to be converted.
            cursor (Cursor): The cursor to Pyodbc connection instance, used to query DB.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query result.
        """
        # Extract column names from the cursor description
        column_names = [desc[0] for desc in cursor.description]

        # Initialize an empty list to store dictionaries
        records_as_dicts = []

        # Iterate over each record in the records
        for record in records:
            # Create a dictionary for the current record
            record_dict = {}
            for i, column_name in enumerate(column_names):
                # Add key-value pairs to the dictionary
                record_dict[column_name] = record[i]

            # Append the dictionary to the list
            records_as_dicts.append(record_dict)

        return records_as_dicts

    def return_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a SQL query on the database and returns the results as a list of dictionaries.

        Args:
            query (str): The SQL query to execute.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query result.
        """
        try:
            # Execute the SQL query and retrieve records and cursor
            records, cursor = self.execute_query(query=query)

            # Log the execution of the SQL query
            _logger.info("Executing SQL query on database")
            _logger.info("Data returned from database: %s", records)

            # Format the query results
            results = self.format_query_results(records=records, cursor=cursor)

            # Close the cursor
            cursor.close()

            return results
        except Exception as e:
            # Log any exceptions that occur
            _logger.error("An error occurred while executing SQL query: %s", e)
            # Raise the exception to be handled at a higher level
            raise

    def save_record(self, insert_query: str) -> None:
        """
        Saves a record to the database.

        Args:
            record (Dict): The record to save to the database.
        """
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(insert_query)
                conn.commit()
                _logger.info("Record saved successfully")
            except Exception as e:
                _logger.error("Error saving record: %s", e)
                conn.rollback()
                self.close(conn)
            finally:
                self.close(conn)
        else:
            _logger.error(DB_CONNECTING_ERROR_MESSAGE)

    def update_record(self, record_id: Any, updated_record: Dict) -> None:
        """
        Updates a record in the database.

        Args:
            record_id (Any): The identifier of the record to update.
            updated_record (Dict): The updated record data.
        """
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()

                # Build the SET clause dynamically based on the keys in updated_record
                set_clause = ", ".join([f"{key} = %s" for key in updated_record.keys()])
                sql = f"UPDATE <TABLE_NAME> SET {set_clause} WHERE id = %s"
                values = list(updated_record.values())
                values.append(record_id)

                cursor.execute(sql, values)
                conn.commit()
                _logger.info("Record updated successfully")
            except Exception as e:
                _logger.error("Error updating record: %s", e)
                conn.rollback()
            finally:
                self.close(conn)
        else:
            _logger.error(DB_CONNECTING_ERROR_MESSAGE)

    def delete_record(self, record_id: Any) -> None:
        """
        Deletes a record from the database.

        Args:
            record_id (Any): The identifier of the record to delete.
        """
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM <TABLE_NAME> WHERE id = %s", (record_id,))
                conn.commit()
                _logger.info("Record deleted successfully")
            except Exception as e:
                _logger.error("Error deleting record: %s", e)
                conn.rollback()
            finally:
                self.close(conn)
        else:
            _logger.error("Error connecting to the database")

    def run_multiple_queries(self, queries: List[str]) -> None:
        """
        Executes multiple SQL queries on the database.

        Args:
            queries (List[str]): A list of SQL queries to be executed.
        """
        conn = self.connect()
        if conn:
            try:
                cursor = conn.cursor()
                for query in queries:
                    cursor.execute(query)
                conn.commit()
                _logger.info("Multiple queries executed successfully")
            except Exception as e:
                _logger.error("Error executing multiple queries: %s", e)
                conn.rollback()
                raise RuntimeError(f"Error executing queries - {query} - {e}")
            finally:
                self.close(conn)
        else:
            _logger.error(DB_CONNECTING_ERROR_MESSAGE)
