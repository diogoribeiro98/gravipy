from .gravdata import *
from .gravmfit import *
from .gcorbits import *
from .utils import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)