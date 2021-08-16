import time
import logging
import random
import socket

from . import config


def get_logger(name, level='INFO', mode='a'):
    logger = logging.getLogger(name)

    fout = logging.FileHandler(config.log_path + name + '.log', mode)

    stdout = logging.StreamHandler()

    if level == "INFO":
        logger.setLevel(logging.INFO)
        fout.setLevel(logging.INFO)
        stdout.setLevel(logging.INFO)
    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        fout.setLevel(logging.DEBUG)
        stdout.setLevel(logging.DEBUG)
    if level == "ERROR":
        logger.setLevel(logging.ERROR)
        fout.setLevel(logging.ERROR)
        stdout.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fout.setFormatter(formatter)
    stdout.setFormatter(formatter)

    logger.addHandler(fout)
    logger.addHandler(stdout)

    return logger


def get_local_ip():
    return socket.gethostbyname(socket.gethostname())


def free_port():
    while True:
        try:
            sock = socket.socket()
            port = random.randint(20000, 65536)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except Exception:
            continue


def time_to_str(t):
    return time.asctime(time.localtime(t))


def float_to_str(f, precision=None):
    if precision is None:
        return '{:g}'.format(f)
    else:
        return '{0:.{1}f}'.format(f, precision)
