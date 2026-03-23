from turtle import pd

import numpy
import pandas
lst = [1, 2, 3, 4, 5]
print(lst)
arr = numpy.array(lst)  
print(arr)
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
df = pandas.DataFrame(data)
pandas.Series([10, 20, 30, 40, 50])
print(df)
print(df['Name'])
print(df['Age'])
print(df['City'])
print(df.describe())
print(df.info())
data = pandas.read_csv("googleplaystore.csv")
print(data['App'])
for i in data.columns:
    print(data[i][1])