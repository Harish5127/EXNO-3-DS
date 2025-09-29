## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:

STEP 1: Read the given Data.

STEP 2: Clean the Data Set using Data Cleaning Process.

STEP 3: Apply Feature Encoding for the feature in the data set.

STEP 4: Apply Feature Transformation for the feature in the data set.

STEP 5: Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

  Developed by : Harish R

  Register no : 212224230085
  
```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="409" height="420" alt="image" src="https://github.com/user-attachments/assets/9070fc9d-6133-40db-892b-d9c15fcb909f" />



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="174" height="218" alt="image" src="https://github.com/user-attachments/assets/6603375b-a01a-455b-bf5d-c3f0c0dcdc53" />



```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="385" height="414" alt="image" src="https://github.com/user-attachments/assets/fa53c8ea-4f92-4abb-9b86-213ac32c3584" />



```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="378" height="425" alt="image" src="https://github.com/user-attachments/assets/d4066a4c-039c-4c64-88fd-1fa8348ceac4" />


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="505" height="412" alt="image" src="https://github.com/user-attachments/assets/5a134f0b-e645-4468-aadf-c2c60b4e1da5" />




```py
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="766" height="420" alt="image" src="https://github.com/user-attachments/assets/bb976b0e-3f23-43b4-9a83-2ed959c65781" />



```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="812" height="424" alt="image" src="https://github.com/user-attachments/assets/095996b2-9fb6-4f14-83a3-6cd9decfc780" />




```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="658" height="429" alt="image" src="https://github.com/user-attachments/assets/b03220ac-ea07-4c69-993b-a4877ad5e825" />



```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="930" height="491" alt="image" src="https://github.com/user-attachments/assets/2d2901d3-5bae-4c3e-bde0-99d2dc470a81" />



```py
df.skew()
```
<img width="343" height="238" alt="image" src="https://github.com/user-attachments/assets/e39b4236-9dc0-4c90-b93a-bc9bce2a9610" />



```py
np.log(df["Highly Positive Skew"])
```
<img width="285" height="544" alt="image" src="https://github.com/user-attachments/assets/103af190-d0c9-43ab-b508-f786e34be55f" />



```py
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="311" height="542" alt="image" src="https://github.com/user-attachments/assets/ac9e6a4e-6940-428a-9134-8c25a81d9e21" />




```py
np.sqrt(df["Highly Positive Skew"])
```
<img width="287" height="545" alt="image" src="https://github.com/user-attachments/assets/3d6b2844-71f3-44a2-9efb-6bf9614c6e3e" />



```py
np.square(df["Highly Positive Skew"])
```
<img width="297" height="544" alt="image" src="https://github.com/user-attachments/assets/eec82187-966a-4339-9677-1f760a584a09" />



```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="927" height="382" alt="image" src="https://github.com/user-attachments/assets/129757d4-d90c-43be-9460-20afc443f61b" />



```py
df.skew()
```
<img width="313" height="254" alt="image" src="https://github.com/user-attachments/assets/b13ef7d5-8ec0-458b-b5b6-8cdf4091ad18" />



```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="379" height="260" alt="image" src="https://github.com/user-attachments/assets/8310b348-ebe2-4283-950f-4dfda460fe82" />


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="983" height="403" alt="image" src="https://github.com/user-attachments/assets/a771abba-a128-4447-87c2-ad45823ab2d2" />


```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="571" height="421" alt="image" src="https://github.com/user-attachments/assets/8982b292-aa91-4e5d-a5b5-39031c970ded" />



```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="548" height="407" alt="image" src="https://github.com/user-attachments/assets/feb56bb5-d8be-4c98-9cf9-8788f2534abc" />




```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="524" height="405" alt="image" src="https://github.com/user-attachments/assets/3488f888-b255-473f-9599-b70a917faa66" />




```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="527" height="402" alt="image" src="https://github.com/user-attachments/assets/55d15b40-545d-434d-b90e-506107eb82e7" />



```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```
<img width="973" height="455" alt="image" src="https://github.com/user-attachments/assets/904073eb-ee68-4a3b-827b-1f89a513a49f" />

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="527" height="402" alt="image" src="https://github.com/user-attachments/assets/25505888-79ec-4539-877d-96071c513587" />


```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="526" height="401" alt="image" src="https://github.com/user-attachments/assets/f4da6dfa-d34f-4ffd-8ea6-c16cef507646" />




  
# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
