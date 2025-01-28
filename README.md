#**Problem Statement**

**Customer Segmentation Analysis:**

- **Description:** Use clustering algorithms to segment customers based on
their purchasing behaviour. This helps in understanding the different
types of customers and tailoring marketing strategies to meet their
needs.
- **Why:** Understanding customer segments allows businesses to target
specific groups with personalized marketing, leading to higher
satisfaction and retention.
- **Tasks:**

    ▪ Collect and preprocess customer data.

    ▪ Example datasets Click Here

    ▪ Apply clustering algorithms (e.g., K-means, hierarchical clustering).

    ▪ Analyse and interpret the clusters.

    ▪ Present findings with visualizations.


**Using Google colab**

from google.colab import files
uploaded = files.upload()

Imported required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

Imported multiple datasets
- dataset 1 -> customer_segmentation_data.csv
- dataset 2 -> Mall_Customers.csv


customer = pd.read_csv('customer_segmentation_data.csv', encoding = 'latin')
mall_customers = pd.read_csv('Mall_Customers.csv')

customer.head()

mall_customers.head()

behavior.head()

**Applying EDA process on each dataset**

###  Dataset -> 1

#dataset 1
customer.info()
customer.describe()

customer.isnull().sum().sum()

customer.duplicated().sum().sum()

**Outlier Analysis**

for i in customer.columns:
  if customer[i].dtypes != "object":
    sns.boxplot(customer[i])
    plt.title(i)
    plt.show()

**Applying label encoding to encode the Gender column data**

le = LabelEncoder()

customer['gender'] = le.fit_transform(customer['gender'])

**Applying one-hot encoding to the preferred_categories data**

encoded_preferred_category = pd.get_dummies(customer['preferred_category'], prefix='preferred_category')
customer = pd.concat([customer, encoded_preferred_category], axis = 1)
customer = customer.drop('preferred_category', axis = 1)

customer.head()

customer.columns

#creating new feature
customer['total_spending'] = customer['spending_score'] * customer['income']

customer.head()

**Standard Scaling**
- Standard Scaling needs to be performed before applying k-means to ensure that all features contribute equally to the distance calculation

#select numerical features for scaling
numerical_features_customer = ['age',	'income',	'spending_score', 'membership_years',	'purchase_frequency',	'last_purchase_amount', 'total_spending']

scaler_customer = StandardScaler()

customer[numerical_features_customer] = scaler_customer.fit_transform(customer[numerical_features_customer])

# **Model Building**

from sklearn.cluster import KMeans

Choosing input column

X = customer[['age',	'gender',	'income',	'spending_score',	'membership_years',	'purchase_frequency',	'last_purchase_amount', 'total_spending']]

wcss = []
for i in range(2,11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++')
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

wcss

sns.set()

#### Plotting the elbow graph

plt.plot(range(1,7), wcss[0:6])
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  #display the plot

from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 11):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)  # Assuming X_data2 is your data
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X, labels))

plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Elbow Method plot
ax1.plot(range(2, 11), wcss)  # Adjust range as needed
ax1.set_title('Elbow Method')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')

# Silhouette Score plot
ax2.plot(range(2, 11), silhouette_scores)  # Adjust range as needed
ax2.set_title('Silhouette Score')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')

plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++') # Initialize KMeans with 6 clusters
kmeans.fit(X) # Fit the model to your data (X_data2)
cluster_label_1 = kmeans.labels_ # Get cluster assignments for each data poi

customer['Cluster'] = cluster_label_1

cluster_summary = customer.groupby('Cluster').mean() # Example: Calculate means
print(cluster_summary)

plt.scatter(customer['income'], customer['spending_score'], c=customer['Cluster'])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation - Dataset 1')
plt.show()

### Dataset -> 2

#dataset 2
mall_customers.info()
mall_customers.describe()

mall_customers.isnull().sum().sum()

mall_customers.duplicated().sum().sum()

**Outlier Analysis for dataset 2**

for i in mall_customers.columns:
  if mall_customers[i].dtypes != "object":
    sns.boxplot(mall_customers[i])
    plt.title(i)
    plt.show()

Treating of Outlier

for i in mall_customers['Annual Income (k$)']:
  Q1 = mall_customers['Annual Income (k$)'].quantile(0.25)
  Q3 = mall_customers['Annual Income (k$)'].quantile(0.75)
  IQR=Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  mall_customers = mall_customers[(mall_customers['Annual Income (k$)'] >= lower_bound) & (mall_customers['Annual Income (k$)'] <= upper_bound)]

Checking for outlier after treating



for i in mall_customers.columns:
  if mall_customers[i].dtypes != "object":
    sns.boxplot(mall_customers[i])
    plt.title(i)
    plt.show()

**applying label encoding for Gender column for dataset 2**

le = LabelEncoder()

mall_customers['Gender'] = le.fit_transform(mall_customers['Gender'])

mall_customers.head()

**Standard Scaling**
- Standard Scaling needs to be performed before applying k-means to ensure that all features contribute equally to the distance calculation

- **Formula ->
`scaled_value = (original_value - mean)/standard_deviation`**

numerical_features_mall = ['Age', 'Annual Income (k$)',	'Spending Score (1-100)']
scaler_mall = StandardScaler()
mall_customers[numerical_features_mall] = scaler_mall.fit_transform(mall_customers[numerical_features_mall])

mall_customers.head()

# **Model Building**

X_data2 = mall_customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

wcss_2 = []

for j in range(2,11):
  kmeans_2 = KMeans(n_clusters = j, init = 'k-means++')
  kmeans_2.fit(X_data2)
  wcss_2.append(kmeans_2.inertia_)

wcss_2

plt.plot(range(2,11), wcss_2[0:10])
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []

for i in range(2, 11):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X_data2)  # Assuming X_data2 is your data
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_data2, labels))

plt.plot(range(2, 11), silhouette_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Elbow Method plot
ax1.plot(range(2, 11), wcss_2)  # Adjust range as needed
ax1.set_title('Elbow Method')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')

# Silhouette Score plot
ax2.plot(range(2, 11), silhouette_scores)  # Adjust range as needed
ax2.set_title('Silhouette Score')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')

plt.show()


kmeans = KMeans(n_clusters=6, init='k-means++') # Initialize KMeans with 6 clusters
kmeans.fit(X_data2) # Fit the model to your data (X_data2)
cluster_labels = kmeans.labels_ # Get cluster assignments for each data point

mall_customers['Cluster'] = cluster_labels # Add a new column to your DataFrame


cluster_summary = mall_customers.groupby('Cluster').mean() # Example: Calculate means
print(cluster_summary)

plt.scatter(mall_customers['Annual Income (k$)'], mall_customers['Spending Score (1-100)'], c=mall_customers['Cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()

