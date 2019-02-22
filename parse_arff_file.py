# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
a = np.array([i for i in range(4)])

from scipy.io import arff
#from io import StringIO

file = r"\\nsq024vs\u7\aczj283\MyDocs\MATLAB\Neural-NetCoursework\PhishingData.arff"


data,meta = arff.loadarff(file)

print(meta._attrnames)


import pandas as pd, numpy as np

df = pd.DataFrame(np.array(data),columns=meta._attrnames).applymap(pd.to_numeric)

df.to_csv(file.replace(".arff",".csv"))


