import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Build a message with the file name and line number where the error happened."""
    _,_,exc_tb = error_detail.exc_info()
    if exc_tb is None:    # raised outside an except block
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in script [{file_name}] at line [{exc_tb.tb_lineno}]: {error}"

class CustomException(Exception):
    """Project-wide exception. Usage: raise CustomException(e, sys)"""
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message

class InvalidInputError(ValueError):
    """The uploaded data cannot be analysed. The message is safe to show to the user."""