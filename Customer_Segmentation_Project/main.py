# ==============================
# 📌 CUSTOMER SEGMENTATION PROJECT
# Using KMeans Clustering (Scikit-learn)
# ==============================

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# ==============================
# 📌 Step 2: Load Dataset
# ==============================
df = pd.read_csv("data/Mall_Customers.csv")

print("\n🔹 Dataset Preview:")
print(df.head())

# ==============================
# 📌 Step 3: Data Cleaning
# ==============================

# Check missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Remove unnecessary column
df.drop("CustomerID", axis=1, inplace=True)

# ==============================
# 📌 Step 4: Encode Categorical Data
# ==============================

# Convert Gender into numbers
df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1})

print("\n🔹 After Encoding:")
print(df.head())

# ==============================
# 📌 Step 5: Feature Selection
# ==============================

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# ==============================
# 📌 Step 6: Elbow Method (Find Best K)
# ==============================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method (Optimal Clusters)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ==============================
# 📌 Step 7: Apply KMeans Model
# ==============================

kmeans = KMeans(n_clusters=5, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)

print("\n🔹 Data with Clusters:")
print(df.head())

# ==============================
# 📌 Step 8: Visualization
# ==============================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Cluster"],
    palette="Set1"
)

plt.title("Customer Segmentation Clusters")
plt.show()

# ==============================
# 📌 Step 9: Save File for Power BI
# ==============================

df.to_csv("data/customer_segments1.csv", index=False)

print("\n✅ File saved successfully for Power BI!")

print(df.columns)
print(df.head())