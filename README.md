# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# EXPLANATION
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# AlGORITHM
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# CODING AND OUPUT
```
REG NO :212223230235
NAME:VASANTHARAJ J
```

```python
import pandas as pd
import seaborn as sns
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
![alt text](<Screenshot 2024-09-10 105514.png>)

```python
df.shape
```
![alt text](<Screenshot 2024-09-10 105537.png>)

```python
df.info()
```
![alt text](<Screenshot 2024-09-10 105552.png>)
```python
df.descibe()
```
![alt text](<Screenshot 2024-09-10 105545.png>)
```python
print(df.head(5))
print("---------------------------")
print(df.tail(5))
```
![alt text](<Screenshot 2024-09-10 105602.png>)
```python
df.isnull().sum()  #df.isna
```
![alt text](<Screenshot 2024-09-10 105609.png>)

```python
df.dropna(how='any').shape
```
![alt text](<Screenshot 2024-09-10 105623.png>)
```python
mn=df.TOTAL.mean()
print(mn)
```
![alt text](<Screenshot 2024-09-10 105636.png>)
```python
df.TOTAL.fillna(mn,inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 105644.png>)
```python
df.M1.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 105922.png>)

```python
df.M2.fillna(method='bfill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 105946.png>)
```python
df.M3.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 110022.png>)
```python
df.M4.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 110049.png>)
```python
df.isna().sum()
```
![alt text](<Screenshot 2024-09-10 110131.png>)
```python
df.duplicated()
```
![alt text](<Screenshot 2024-09-10 110138.png>)
```python
df.drop_duplicates(inplace=True)
df.duplicated()
```
![alt text](<Screenshot 2024-09-10 110153.png>)
```python
df.DOB
```
![alt text](<Screenshot 2024-09-10 110214-1.png>)
```python
df['DOB']= pd.to_datetime(df['DOB'],format='%Y.%m.%d',errors='coerce')
df
```
![alt text](<Screenshot 2024-09-10 110226.png>)

```python
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2024-09-10 110250.png>)
```python
df['DOB'].fillna(method='bfill',inplace=True)
df['M4'].fillna(method='bfill',inplace=True)
df['AVG'].fillna(method='bfill',inplace=True)
df['TOTAL'].fillna(method='bfill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 110312.png>)
```python
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2024-09-10 110321.png>)

# OUTLIER DETECTION AND REMOVAL USING IQR
```python
import pandas as pd
import seaborn as sns
import numpy as np
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
df=pd.DataFrame(age)
df
```
![alt text](<Screenshot 2024-09-10 110331.png>)
```python
sns.boxplot(df)
```
![alt text](<Screenshot 2024-09-10 110344.png>)
```python
sns.scatterplot(data=df)
```
![alt text](<Screenshot 2024-09-10 110357.png>)
```python
q1=df.quantile(0.25)
q2=df.quantile(0.5)
q3=df.quantile(0.75)
iqr=q3-q1
iqr
```
![alt text](<Screenshot 2024-09-10 110405.png>)
```python
Q1=np.percentile(df,25)
Q3=np.percentile(df,75)
IQR=Q3-Q1
IQR
```
![alt text](<Screenshot 2024-09-10 110411.png>)
```python
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
print(lower_bound)
print(upper_bound)
```
![alt text](<Screenshot 2024-09-10 110417.png>)
```python
outliers=[x for x in age if x<lower_bound or x>upper_bound]
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower Bound:",lower_bound)
print("Upper Bound:",upper_bound)
print("Outliers:",outliers)
```
![alt text](<Screenshot 2024-09-10 110430.png>)
```python
df=df[((df>=lower_bound)&(df<=upper_bound))]
df
```
![alt text](<Screenshot 2024-09-10 110445.png>)
```python
df=df.dropna()
df
```
![alt text](<Screenshot 2024-09-10 110451.png>)
```python
sns.boxplot(data=df)
```
![alt text](<Screenshot 2024-09-10 110507.png>)
```python
sns.scatterplot(data=df)
```
![alt text](<Screenshot 2024-09-10 110514.png>)
```python
data=[1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean=np.mean(data)
std=np.std(data)
print('mean of the dataset is',mean)
print('std.deviation is',std)
```
![alt text](<Screenshot 2024-09-10 110520.png>)
```python
threshold=3
outlier=[]
for i in data:
  z=(i-mean)/std
  if z>threshold:
    outlier.append(i)
print('outlier in dataset is',outlier)
```
![alt text](<Screenshot 2024-09-10 110530.png>)
```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
data={'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,
                66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df=pd.DataFrame(data)
df
```
![alt text](<Screenshot 2024-09-10 110633.png>)
![alt text](<Screenshot 2024-09-10 110639.png>)
```python
z=np.abs(stats.zscore(df))
print(df[z['weight']>3])
```
![alt text](<Screenshot 2024-09-10 110653.png>)








# RESULT
Thus we have cleared the data and removed the outlier by detection using IQR and Z-score method
