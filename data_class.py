"""
File Metadata Model Module

This module provides a structured representation of file metadata within the storage
and processing system. It defines the core data structures for tracking file
attributes, history, and relationships throughout the processing pipeline.

Key Components:
    FileMetadata: Dataclass representing comprehensive file metadata
        - Tracks file identification and relationships
        - Stores temporal information and file attributes
        - Maintains content classification data
        - Supports versioning of metadata schema

Data Validation Rules:
    - All string fields must be non-empty unless explicitly noted
    - Timestamps must follow ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
    - File sizes must be non-negative integers
    - File paths must use forward slashes (/) as separators
    - Extensions should be lowercase without leading dots
    - IDs should be globally unique within the system

Usage Example:
    >>> from file_metadata import FileMetadata
    >>> 
    >>> # Create a metadata instance for a PDF document
    >>> document_metadata = FileMetadata(
    ...     file_id="doc_123abc",
    ...     parent_id="folder_456def",
    ...     filename="quarterly_report.pdf",
    ...     full_path="/reports/2024/Q1/quarterly_report.pdf",
    ...     category="financial_document",
    ...     size_bytes=2048576,
    ...     created_at="2024-03-21T10:00:00Z",
    ...     modified_at="2024-03-21T10:00:00Z",
    ...     created_by="user_789ghi",
    ...     content_type="document",
    ...     extension="pdf",
    ...     mime_type="application/pdf",
    ...     metadata_version="1.0"
    ... )
    >>> 
    >>> # Access metadata attributes
    >>> print(f"File: {document_metadata.filename}")
    >>> print(f"Size: {document_metadata.size_bytes} bytes")
    >>> print(f"Path: {document_metadata.full_path}")

Integration Points:
    - Database storage through DatabaseManager
    - File processing pipeline inputs/outputs
    - API response serialization
    - Metadata validation services
    - Search indexing systems

Best Practices:
    1. Always validate timestamps before creating instances
    2. Use consistent casing for extensions and categories
    3. Generate globally unique IDs for file_id
    4. Maintain parent_id relationships carefully
    5. Follow system-defined content type categories

Dependencies:
    - dataclasses: For dataclass decorator and functionality
    - datetime (optional): For timestamp validation
    - typing (optional): For type hints
    - uuid (optional): For ID generation

Notes:
    - All string fields are case-sensitive unless noted
    - Timezone information is required for timestamps
    - Parent IDs should be validated against existing containers
    - MIME types should follow standard formats
    - Extensions should match actual file contents
    
TODO:
    - 

FIXME:
    - 
"""

from dataclasses import dataclass


@dataclass
class FileMetadata:
    """
    Data class representing metadata for processed files.

    This class encapsulates all relevant metadata information for files processed
    within the system, including identification, hierarchical organization,
    temporal information, and content classification.

    Attributes:
        file_id (str): Unique identifier for the file
        parent_id (str): Identifier for the parent container/folder
        filename (str): Name of the file including extension
        full_path (str): Complete path to the file in the system
        category (str): Classification category for the file
        size_bytes (int): Size of the file in bytes
        created_at (str): ISO timestamp of file creation
        modified_at (str): ISO timestamp of last modification
        created_by (str): Identifier of the user/system that created the file
        content_type (str): High-level content type classification
        extension (str): File extension without leading dot
        mime_type (str): MIME type of the file
        metadata_version (str): Version of the metadata schema, defaults to "1.0"

    Example:
        >>> file_meta = FileMetadata(
        ...     file_id="8g3498siufi094f90h",
        ...     parent_id="3478fg837g83874f",
        ...     filename="document.pdf",
        ...     full_path="/documents/2024/document.pdf",
        ...     category="documents",
        ...     size_bytes=1048576,
        ...     created_at="2024-01-01T10:00:00Z",
        ...     modified_at="2024-01-01T10:00:00Z",
        ...     created_by="user123",
        ...     content_type="document",
        ...     extension="pdf",
        ...     mime_type="application/pdf"
        ... )

    Notes:
        - All timestamp fields should be in ISO   format
        - The file_id and parent_id should be system-generated unique identifiers
        - The mime_type should follow the standard MIME type format
    """

    file_id: str
    parent_id: str
    filename: str
    full_path: str
    category: str
    size_bytes: int
    created_at: str
    modified_at: str
    created_by: str
    content_type: str
    extension: str
    mime_type: str
    metadata_version: str = "1.0"