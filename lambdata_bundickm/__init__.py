#!/usr/bin/env python
"""
lambdata - collection of Data Science helper functions
"""

import pandas as pd
import numpy as np
from . import example_module

y = example_module.increment(example_module.x)
TEST = pd.DataFrame(np.ones(10))
