import logging
import numpy as np
from astropy.time import Time
from datetime import datetime

color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'

log_level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
   
def fiber_coupling(x):
    fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(x**2))/280)**2)
    return fiber_coup

def convert_date(date, mjd=False):
    t = Time(date)
    if mjd:
        return t.mjd
    t2 = Time('2000-01-01T12:00:00')
    date_decimal = (t.mjd - t2.mjd)/365.25+2000
    date = date.replace('T', ' ')
    date = date.split('.')[0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date_decimal, date


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def timing(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger = logging.getLogger(__name__)
        logger.info(f'Function {func.__name__} execution time: {end-start:.2f} s')
        return result
    return wrapper