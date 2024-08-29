# EX NO:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# EXPLANATION
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# ALGORITHM
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# CODE
## data cleaning


import pandas as pd
df=pd.read_csv('/content/SAMPLEIDS.csv')
df

![image](https://github.com/user-attachments/assets/f01f82b6-b3c7-448e-937e-f5a31c3a67d8)


df.isnull()

![image](https://github.com/user-attachments/assets/0bcaceb1-5541-4716-ae5a-030ca417338b)


df.notnull()

![image](https://github.com/user-attachments/assets/e8d5f5b8-623d-4c1d-ac69-7c5b5ba71377)


df.dropna(axis=1)

![image](https://github.com/user-attachments/assets/5d709f82-aa53-4a38-8736-bf88364c330a)


df.fillna(0)

![image](https://github.com/user-attachments/assets/7d39380e-2036-4f93-a8a1-66645077e7f1)


df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})

![image](https://github.com/user-attachments/assets/5bcb0a83-e080-48ab-aabf-128b0591a85f)

## IQR(Inter Quartile Range)

import pandas as pd
ir=pd.read_csv('/content/iris.csv')
ir

![image](https://github.com/user-attachments/assets/23a67cf1-2594-47d0-991f-23741eb2c89e)


ir.describe()

![image](https://github.com/user-attachments/assets/0efa4591-af49-4a56-8704-4786abbce157)


import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)

![image](https://github.com/user-attachments/assets/c2d60711-428d-4b75-b331-28c5025471d2)


c1=ir.sepal_width.quantile(0.25)
c3=ir.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)

![image](https://github.com/user-attachments/assets/18eacd98-116e-4678-8b13-ed70571ef75e)


rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']

![image](https://github.com/user-attachments/assets/7b5f479d-c8ea-48e2-aec3-0cd3af183ffa)


delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid

![image](https://github.com/user-attachments/assets/12e2045f-2264-4072-ace6-355ffb1d93f7)


sns.boxplot(x='sepal_width',data=delid)

![image](https://github.com/user-attachments/assets/d465ad5e-59b7-4671-a755-f4d40631c934)

## Z-SCORE


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
dataset=pd.read_csv("heights.csv")
dataset            

![image](https://github.com/user-attachments/assets/8b16190c-7102-499c-b962-28225dd8d9ab)


df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr

![image](https://github.com/user-attachments/assets/2de0ba89-f817-411a-9415-650557714d28)


low = q1 - 1.5*iqr
print(low)
high = q3 + 1.5*iqr
print(high)

![image](https://github.com/user-attachments/assets/900983bc-21b4-4665-a67a-2e5bf4a89af8)


df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1

![image](https://github.com/user-attachments/assets/6b3b5010-62a9-4967-bd75-c17bd3bec2f5)


z = np.abs(stats.zscore(df['height']))
z

![image](https://github.com/user-attachments/assets/42fdad42-7999-429c-929a-c62767b4d939)


df1 = df[z<3]
df1

![image](https://github.com/user-attachments/assets/7b7298a0-b391-40f8-b605-a39a1d704a90)


# RESULT:
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
