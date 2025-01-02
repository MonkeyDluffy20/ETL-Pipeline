

## Introduction

Welcome to ETL-Pipeline, a comprehensive solution integrating Azure Blob Storage, Pinecone, and SQL Server to streamline file metadata extraction, storage, and retrieval. This system automates the processing of diverse file formats, leverages Pinecone for fast and efficient vector-based querying, and SQL Server for centralized metadata management, enabling teams to handle and analyze data effortlessly.

## Features

- ### File Metadata Extraction

  Automatically fetch and extract metadata from files stored in Azure Blob Storage, including file name, size, type, and timestamps.

- ### Metadata Storage

  Stores extracted metadata in SQL Server for centralized management and utilizes Pinecone for fast similarity-based querying and efficient analysis.

- ### Multi-File Format Support

  Processes a wide range of file formats, including JSON, PDF, and XLSX, with custom loaders tailored to each format.

- ### Scalable Processing

  Handles batch file processing using multithreading for optimized performance and minimal downtime.

- ### Namespace Management with Pinecone

  Organizes data into distinct namespaces within a single index, ensuring logical separation for different collections while optimizing query performance.

- ### API Integration

  Seamlessly integrates with APIs for Pinecone and Azure Blob Storage, facilitating efficient data retrieval, processing, and storage.

- ### Logging and Error Handling

  Provides comprehensive logging for all operations and robust error handling to ensure reliable and maintainable workflows.

## Prerequisites

Ensure the following software is installed on your system:

- Python 3.8 or higher
- pip (Python package manager)

## Run Locally

Clone the project:

```bash
  git clone the repo
```

Go to the project directory:

```bash
  cd ETL-pipeline
```

Set up a virtual environment:

```bash
  python -m venv .venv
  source .venv/bin/activate # on bash
  .venv\Scripts\activate # on Windows
```

Install dependencies:

```bash
  pip install -r requirements.txt
```

## Environment Variables

To run this project, add the following environment variables to your `.env` file:

- ### Azure Blob Storage Configuration Definitions

  `BLOB_STORAGE_CONNECTION_STRING`: Connection string used to access Azure Blob Storage.

  `BLOB_CONTAINER_NAME`: Name of the container within Azure Blob Storage where files are stored.

  `BLOB_STORAGE_ACCOUNT_NAME`: The name of the Azure Blob Storage account.

- ### SQL Server Configuration Definitions

  `SQL_SERVER`: Hostname or IP address of the SQL Server instance.

  `SQL_DATABASE` (optional): Database name for SQL Server.

  `SQL_USER` (optional): Username for SQL Server authentication (if not using a trusted connection).

  `SQL_PASSWORD` (optional): Password for SQL Server authentication (if not using a trusted connection).

## Explanation

These environment variables configure the project's connections to Azure Blob Storage, SQL Server, and Azure Key Vault. They ensure secure authentication and authorization while managing sensitive data like API keys, database credentials, and file storage.

**Note**: Always store `.env` files securely and avoid sharing them publicly to prevent unauthorized access to your resources.

## Configure Environment Variables

Create a `.env` file in the project root and add the following environment variables:

### Azure Blob Storage Configuration

```bash
BLOB_STORAGE_CONNECTION_STRING="your_connection_string"
BLOB_CONTAINER_NAME="test-container"
BLOB_STORAGE_ACCOUNT_NAME="storage"
```

### SQL Server Configuration

```bash
SQL_SERVER=your_sql_server
SQL_DATABASE=your_database
```

## ETL-Pipeline Architecture

The ETL pipeline processes files stored in Azure Blob Storage and extracts metadata, which is stored both in SQL Server for centralized management and in Pinecone for advanced similarity-based querying. The pipeline organizes data into namespaces within a single Pinecone index (`BBGD-Index`), ensuring logical separation for different collections while maintaining high query efficiency.

### Key Steps:

1. **Extract**: Fetch metadata from Azure Blob Storage.
2. **Transform**: Prepare metadata for efficient storage and querying.
3. **Load**:
   - Store metadata in SQL Server for relational data handling.
   - Store vectorized data in Pinecone, categorized into namespaces based on file types or other criteria.
