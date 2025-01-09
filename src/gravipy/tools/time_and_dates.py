import logging
from functools import wraps
from datetime import datetime
from astropy.time import Time

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger = logging.getLogger(__name__)
        logger.info(f'Function {func.__name__} execution time: {end-start:.2f} s')
        return result
    return wrapper


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


def print_status(number, total):
    number = number+1
    if number == total:
        print("\rComplete: 100%")
    else:
        percentage = int((number/total)*100)
        print("\rComplete: ", percentage, "%", end="")