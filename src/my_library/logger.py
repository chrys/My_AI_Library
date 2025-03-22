import logging
import os

def setup_logger(level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = '/var/log/my_chatbot'
    log_file = os.path.join(log_dir, 'my_library.log')
    
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create a file handler
    try:
        handler = logging.FileHandler(log_file)
    except PermissionError:
        # Fallback to a temporary directory if production directory is not accessible
        import tempfile
        temp_dir = tempfile.gettempdir()
        log_file = os.path.join(temp_dir, 'my_library.log')
        handler = logging.FileHandler(log_file)
    
    handler.setLevel(level)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Remove existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

def set_logging_level(logger, level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)