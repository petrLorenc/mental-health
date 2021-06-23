import sys
import logging

logger = logging.getLogger('training')
ch = logging.StreamHandler(sys.stdout)
# create formatter
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)