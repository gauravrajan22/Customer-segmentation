X2=df.loc[:,["Annual Income (k$)","Spending Score (1-100)"]].values
wcss1=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(X2)
    wcss1.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss1,linewidth=2,color="red",marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS1")
plt.show()