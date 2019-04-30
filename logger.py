import logging
from logging.handlers import RotatingFileHandler
import gasprognoseConstants

class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.handler = logging.handlers.RotatingFileHandler(gasprognoseConstants.LOGFILE, 'a',
                                                            maxBytes=128 * 1024, backupCount=1)
        self.handler.terminator = '\r\n'
        self.formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(filename)s %(funcName)s'
                                           ' %(lineno)d:  %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
