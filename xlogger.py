import datetime
import logging as log

log.basicConfig(filename='predict.log', level=log.INFO, format='%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')


def log(m):
    log.warning(m)
    print(f"{datetime.datetime.now()} {m}")
