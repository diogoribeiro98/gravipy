import json
import logging
from pkg_resources import resource_filename

log_level_mapping = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def setup_logging():
    """Setup logging format and handlers according to the json file
    """
    config_file = resource_filename(__name__, 'config.json')
    
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)

