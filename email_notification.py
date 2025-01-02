# Add directories to the system path for custom module imports
import sys

sys.path.append("./")
sys.path.append("../")

# Import Dependencies
import os
import base64
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)

from typing import List, Optional

# Configure logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmailSender:
    """
    A class to send an email with a CSV attachment created from a pandas DataFrame using SendGrid.

    ``` python

    # Create an instance of CSVEmailSender
    sender = EmailSender(
        from_email='from_email@example.com',
        to_email=to@example.com',
        subject='Sending with Twilio SendGrid is Fun',
        html_content='<strong>and easy to do anywhere, even with Python</strong>',
        cc_emails=['cc@example.com'],
        bcc_emails=['bcc@example.com'],
        file_path='data.csv',
        file_type='text/csv'
    )

    # Send the email with the DataFrame as a CSV attachment
    status_code, body, headers = sender.send_email()
    ```
    """

    def __init__(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_content: str,
        cc_emails: Optional[List[str]] = None,
        bcc_emails: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
    ):
        """
        Initialize the CSVEmailSender with email details and SendGrid API key.

        Args:
            from_email (str): Sender's email address.
            to_email (str): Recipient's email address.
            subject (str): Subject of the email.
            html_content (str): HTML content of the email.
            cc_emails (Optional[List[str]]): List of CC email addresses.
            bcc_emails (Optional[List[str]]): List of BCC email addresses.
            file_path (Optional[str]): Path to the file to be attached.
            file_type (Optional[str]): MIME type of the file.
        """
        self.from_email = from_email
        self.to_emails = to_email
        self.subject = subject
        self.html_content = html_content
        self.cc_emails = cc_emails
        self.bcc_emails = bcc_emails
        self.file_path = file_path
        self.file_type = file_type

    def create_attachment(
        self, file_path: str, file_type: str, disposition: str = "attachment"
    ) -> Attachment:
        """
        Create a SendGrid Attachment object from a file.

        Args:
            file_path (str): Path to the file to be attached.
            file_type (str): MIME type of the file.
            disposition (str): Disposition of the file (default is 'attachment').

        Returns:
            Attachment: SendGrid Attachment object.
        """
        try:
            with open(file_path, "rb") as f:
                file_content = base64.b64encode(f.read()).decode("utf-8")
            file_name = os.path.basename(file_path)
            attachment = Attachment(
                FileContent(file_content),
                FileName(file_name),
                FileType(file_type),
                Disposition(disposition),
            )
            _logger.info(f"Attachment created successfully from file: {file_path}")
            return attachment
        except Exception as e:
            _logger.error(f"Error creating attachment from file: {e}")
            raise

    def send_mail(self):
        """
        Send an email with a CSV attachment using SendGrid.

        Returns:
            Tuple[int, str, Dict[str, str]]: Status code, response body, and response headers.
        """
        try:
            # Create a SendGrid API client
            sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))

            # Create a Mail object
            message = Mail(
                from_email=self.from_email,
                to_emails=self.to_emails,
                subject=self.subject,
                html_content=self.html_content,
            )

            # Add CC and BCC emails if provided
            if self.cc_emails:
                message.cc = self.cc_emails
            if self.bcc_emails:
                message.bcc = self.bcc_emails

            # Validate file_path and file_type consistency
            if (self.file_path and not self.file_type) or (
                self.file_type and not self.file_path
            ):
                raise ValueError(
                    "Both file path and file type must be provided together or not at all."
                )

            # Add the CSV attachment to the email
            if self.file_path and self.file_type:
                attachment = self.create_attachment(
                    file_path=self.file_path,
                    file_type=self.file_type,
                )
                message.attachment = attachment

            # Send the email
            response = sg.send(message)
            _logger.info(f"Email sent successfully to: {self.to_emails}")
            return response.status_code, response.body, response.headers
        except Exception as e:
            _logger.error(f"Error sending email: {e}")
            raise
