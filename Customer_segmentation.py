import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Read the data from a CSV file
df = pd.read_csv("Data.csv")

# Print the first few rows of the DataFrame
print(df.head())

# Print the column names of the DataFrame
print(df.keys())

# Get descriptive statistics of the DataFrame
print(df.describe())

# Remove the 'CustomerID' column from the DataFrame
df.drop(["CustomerID"], axis=1, inplace=True)

# Plot a histogram for the 'Age' column
sns.histplot(data=df['Age'], bins=20, kde=True, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# Plot a histogram for the 'Annual Income' column
sns.histplot(data=df['Annual Income (k$)'], bins=20, kde=True, edgecolor='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.title('Histogram of Annual Income (k$)')
plt.show()

# Plot a histogram for the 'Spending Score' column
sns.histplot(data=df['Spending Score (1-100)'], bins=20, kde=True, edgecolor='black')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Histogram of Spending Score (1-100)')
plt.show()

# Count the number of customers by gender and create a bar plot
plt.figure(figsize=(15, 5))
sns.countplot(y="Gender", data=df)
plt.show()

# Create violin plots for each numerical column by gender
plt.figure(1, figsize=(15, 7))
n = 0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=cols, y="Gender", data=df)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title("Violin Plot")
plt.show()

# Split the 'Age' column into different age groups and count the number of customers in each group
age_18_25 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
age_26_35 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
age_36_45 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
age_46_55 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
age_55above = df.Age[df.Age >= 56]

agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
agey = [len(age_18_25.values), len(age_26_35.values), len(age_36_45.values), len(age_46_55.values),
        len(age_55above.values)]

# Create a bar plot to show the number of customers in each age group
plt.figure(figsize=(15, 6))
sns.barplot(x=agex, y=agey, palette="mako")
plt.title("Number of Customers and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# Split the 'Spending Score' column into different score ranges and count the number of customers in each range
ss_1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss_21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss_41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss_61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss_81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss_1_20.values), len(ss_21_40.values), len(ss_41_60.values), len(ss_61_80.values), len(ss_81_100.values)]

# Create a bar plot to show the distribution of spending scores
plt.figure(figsize=(15, 6))
sns.barplot(x=ssx, y=ssy, palette="rocket")
plt.title("Spending Scores of Customers")
plt.xlabel("Score")
plt.ylabel("Number of Customers")
plt.show()

# Split the 'Annual Income' column into different income ranges and count the number of customers in each range
ai10_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

aix = ["$ 0-30,000", "$ 30,001-60,000", "$ 60,001-90,000", "$ 90,001-1,20,000", "$ 1,20,001-1,50,000"]
aiy = [len(ai10_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

# Create a bar plot to show the distribution of annual incomes
plt.figure(figsize=(15, 6))
sns.barplot(x=aix, y=aiy, palette="Spectral")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customers")
plt.show()

# Cluster the data based on Age and Spending Score using K-means
X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values
wcss = []  # Within-Cluster Sum of Squares (WCSS)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.title("Elbow Curve for Age and Spending Score")
plt.show()

# Perform K-means clustering on Age and Spending Score with k=4
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X1)
print(labels)
print(kmeans.cluster_centers_)

# Visualize the clusters
plt.scatter(X1[:, 0], X1[:, 1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black")
plt.title("Clusters of Customers (Age vs Spending Score)")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Cluster the data based on Annual Income and Spending Score using K-means
X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
wcss1 = []  # Within-Cluster Sum of Squares (WCSS)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss1.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss1, linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS1")
plt.title("Elbow Curve for Annual Income and Spending Score")
plt.show()

# Perform K-means clustering on Annual Income and Spending Score with k=5
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X2)
print(labels)
print(kmeans.cluster_centers_)

# Visualize the clusters
plt.scatter(X2[:, 0], X2[:, 1], c=kmeans.labels_, cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black")
plt.title("Clusters of Customers (Annual Income vs Spending Score)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

X3=df.iloc[:,1:]
wcss2=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(X3)
    wcss2.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss2,linewidth=2,color="red",marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS2")
plt.title("Elbow Curve for All Features")
plt.show()

kmeans=KMeans(n_clusters=5)
label=kmeans.fit_predict(X3)
print(label)
print(kmeans.cluster_centers_)

clusters=kmeans.fit_predict(X3)
df["label"]=clusters

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(df.Age[df.label==0],df["Annual Income (k$)"][df.label==0],df["Spending Score (1-100)"]
           [df.label==0],c="blue",s=60)
ax.scatter(df.Age[df.label==1],df["Annual Income (k$)"][df.label==1],df["Spending Score (1-100)"]
           [df.label==1],c="red",s=60)
ax.scatter(df.Age[df.label==2],df["Annual Income (k$)"][df.label==2],df["Spending Score (1-100)"]
           [df.label==2],c="green",s=60)
ax.scatter(df.Age[df.label==3],df["Annual Income (k$)"][df.label==3],df["Spending Score (1-100)"]
           [df.label==3],c="orange",s=60)
ax.scatter(df.Age[df.label==4],df["Annual Income (k$)"][df.label==4],df["Spending Score (1-100)"]
           [df.label==4],c="purple",s=60)
ax.view_init(30,185)
plt.title("Clusters of Customers (All Features)")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()



