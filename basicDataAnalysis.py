# import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# loading the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species row
df['species'] = iris.target_names[iris.target]

# Display the first 8 rows of the dataset
print(df.head())

#Explore the structure by checking data types and any missing values
# Data types
result = df.dtypes
print(result)

# Missing values
missing_values = df.isnull()
print(missing_values)

# clean dataset
# df.fillna(df.mean(), inplace=True)

# compute basic statistics using .describe()
summary_stats = df.describe()
print(summary_stats)

# grouping by species to perform mean of sepal width and length, and petal width and length
grouped = df.groupby("species").agg({'sepal length (cm)': 'mean',
                                     'sepal width (cm)': 'mean',
                                     'petal width (cm)': 'mean',
                                     'petal length (cm)': 'mean'})
print(grouped)

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# line chart
plt.plot(df["species"], df["petal length (cm)"], label="species")
plt.title("Species to petal length")
plt.xlabel("species")
plt.ylabel("petal length (cm)")
plt.legend()
plt.show()

# bar chart
sns.barplot(x='species', y='petal length (cm)', color='red', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram
plt.hist(df['sepal length (cm)'], bins=20, edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.ylabel('Frequency')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

#Error handling
try:
    df = pd.read_csv("iris.csv")
except FileNotFoundError:
    print("The file was not found.")
