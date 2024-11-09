import sys

import sys
import traceback

# Function to generate a detailed error message with additional context
def error_message_detail(error, error_detail: sys, hint=None):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    function_name = exc_tb.tb_frame.f_code.co_name
    error_type = type(error).__name__

    # Base error message with file, line number, and error type
    error_message = (
        f"Error occurred in script [{file_name}], "
        f"function [{function_name}], line [{exc_tb.tb_lineno}]: "
        f"[{error_type}] {error}"
    )
    
    # Add hint if available
    if hint:
        error_message += f" | Hint: {hint}"

    return error_message

# Custom Exception class with optional logging level and hints
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys, hint=None, level="ERROR"):
        # Generate the detailed error message
        detailed_message = error_message_detail(error_message, error_detail=error_detail, hint=hint)
        super().__init__(detailed_message)
        self.error_message = detailed_message
        self.level = level  # Add logging level attribute (e.g., ERROR, WARNING)

    def __str__(self):
        # Display the error message with level if specified
        return f"[{self.level}] {self.error_message}"
