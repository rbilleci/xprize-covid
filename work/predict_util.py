
from datetime import timedelta


def date_range(start_date, end_date, include_end_date):
    days = int((end_date - start_date).days)
    if include_end_date:
        days = days + 1
    for n in range(int(days)):
        yield start_date + timedelta(n)

