import os
import re

# years as currently available in dataset
_MIN_YEAR = 1880
_MAX_YEAR = int(re.search('^yob([0-9]{4}).txt$', os.listdir('data/names/')[-1]).group(1))
