import pandas as pd
from sklearn import preprocessing
import numpy as np
cc= pd.read_csv("/users/jing/documents/callcenterdataset.csv")
cc_star = preprocessing.MinMaxScaler(cc)
print(cc_star)