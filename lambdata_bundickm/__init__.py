#!/usr/bin/env python
"""
lambdata - collection of Data Science helper functions
"""
#some of these imports are for functions I haven't added yet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from matplotlib.colors import LinearSegmentedColormap
from tabulate import tabulate
from . import eda

TEST = pd.DataFrame(np.ones(10))
