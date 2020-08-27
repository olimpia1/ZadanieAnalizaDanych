import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
1. Czy istnieje zależność pomiędzy tym, kto jest bezpośrednim przełożonym (ManagerName, ManagerID) danego pracownika,
 a oceną wydajności pracy (PerformanceScore)?
"""

df = pd.read_csv('HRDataset.csv')
df.drop(['LastPerformanceReview_Date', 'DaysLateLast30'], axis=1, inplace=True)
df.dropna(thresh=2, inplace=True)
df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%Y')

df['DateofHire'] = pd.to_datetime(df['DateofHire'], format='%m/%d/%Y')
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'], format='%m/%d/%Y')


def numerize_performance(row):
    if row['PerformanceScore'].startswith("Fully"):

        return 3.1
    elif row['PerformanceScore'].startswith("Needs"):

        return 1
    elif row['PerformanceScore'].startswith("Exceeds"):

        return 4.3
    elif row['PerformanceScore'].startswith("PIP"):

        return 2


df['Satisfaction'] = df.apply(lambda row: numerize_performance(row), axis=1)
plt.figure(figsize=(45, 10))
sns.barplot(x='ManagerName', y='Satisfaction', data=df).get_figure().savefig("output1.png")
plt.clf()


"""
2. Jakie źródła pozyskania pracownika (RecruitmentSource) są najlepsze, 
   jeśli zależy nam na jak najdłuższym stażu pracowników?
"""


def count_seniority(row):
    if pd.isnull(row['DateofTermination']):
        end_date = dt.datetime(2019, 9, 27)
    else:
        end_date = row['DateofTermination']
    return (end_date - row['DateofHire'])/np.timedelta64(1, 'Y')


df['Seniority'] = df.apply(lambda row: count_seniority(row), axis=1)
plt.clf()
plt.figure(figsize=(45,10))
sns.barplot(x='RecruitmentSource', y='Seniority', data=df).get_figure().savefig("output2.png")
plt.clf()


"""
3.Czy stan cywilny (MaritalDesc) pracownika koreluje w jakikolwiek sposób z zadowoleniem z pracy (EmpSatisfaction)?
"""

sns.boxplot(x='MaritalDesc', y='EmpSatisfaction', data=df).get_figure().savefig("output3.png")
plt.clf()

"""
4. Jak wygląda struktura wieku aktualnie zatrudnionych pracowników?
"""


def count_age(row):
    end_date = dt.datetime(2019, 9, 27)

    return (end_date - (row['DOB']))/np.timedelta64(1,'Y')


df['Age'] = df.apply(lambda row: count_age(row), axis=1)
sns.distplot(df['Age']).get_figure().savefig("output4.png")
plt.clf()

"""
5. Czy starsi pracownicy pracują nad większą liczbą specjalnych projektów niż młodsi pracownicy?
"""

sns.jointplot(x='Age', y='SpecialProjectsCount', data=df, kind="regr").savefig("output5.png")
plt.clf()
df_1 = df[['Age', 'SpecialProjectsCount']]
sns.pairplot(df_1).savefig("output6.png")
