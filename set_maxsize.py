"""
this module sets the max length the CSV parser can support
this way we can have lots of text data to train with
"""

import csv
import sys


def set_csv_max_length():
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
