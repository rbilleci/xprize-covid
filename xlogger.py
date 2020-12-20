import datetime
from logging import basicConfig, warning, INFO

basicConfig(filename='predict.log', level=INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')


def log(m):
    warning(m)
    print(f"{datetime.datetime.now()} {m}")
