# import numpy as np
# import pandas as pd
# import os

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.impute import SimpleImputer

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         data_path = os.path.join(dirname, filename)
#         data_set = pd.read_csv(data_path) # load
#         data_set = data_set.drop(["gender", "race/ethnicity", "lunch", "parental level of education", "test preparation course"], axis=1) # remove text nominal data
# #         data_set = data_set.describe() #Generate descriptive statistics.
# #         print(data_set) #exploratory data analysis. 
#         a = data_set.iloc[:, :-1].values
#         b = data_set.iloc[:, 1].values

#         imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#         imputer = imputer.fit(a[:, 1:])
#         a[:, 1:] = imputer.transform(a[:, 1:])
# #         print(b)

# def main():
#     pass

# def clean():
#     pass
# def reduce():
#     pass
# def transform():
#     pass
# def integrate():
#     pass

import numpy as np
import pandas as pd
    
import os
TO_DROP_COLUMNS = ["gender", "race/ethnicity", "lunch", "parental level of education", "test preparation course"]
# print(TO_DROP_COLUMNS)

from sklearn.impute import SimpleImputer
fill = SimpleImputer(missing_values=np.nan, strategy='mean')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_path = os.path.join(dirname, filename)
        data_set = pd.read_csv(data_path) # load
        data_set = data_set.drop(TO_DROP_COLUMNS, axis=1)
#         data_set = data_set.describe() #Generate descriptive statistics.
#         print(data_set.head(10))
        
# #         print(data_set) #exploratory data analysis. 
#         b = data_set.iloc[:, 1].values
        a = data_set.iloc[:].values
        c = fill.fit(a)
        a[:] = c.transform(a[:])

        print(data_set.head(10))

