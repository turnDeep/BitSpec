import logging
import sys

def setup_logger(name='ei-ms-predictor', level=logging.INFO):
    """
    Sets up a basic logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Create a handler
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

# Default logger
log = setup_logger()
