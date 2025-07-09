import logging
import os




# Configure the logging
def setup_logger(level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create a file handler
    handler = logging.FileHandler('my_library.log')
    handler.setLevel(level)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

# Example of how to change the logging level
def set_logging_level(logger, level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

# Usage
#if __name__ == "__main__":
#    logger = setup_logger()
#    logger.info("Logger is set up.")