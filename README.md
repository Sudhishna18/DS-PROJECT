# DS-PROJECT-SALES DATA ANALYSIS
## AIM:
To clean, analyze, and visualize the sales dataset to identify key trends and insights using data preprocessing, EDA, feature selection, and visualization techniques.
## NAME: P.SUDHISHNA
## REG NO: 212224040336

## Objective

* Clean and organize the accident dataset
* Handle missing or duplicate records
* Fix inconsistent or unrealistic values
* Encode and scale features for model readiness
* Generate visual insights into patterns and correlations

## Dataset Information

The dataset used in this project contains retail sales records collected from multiple cities. It includes 185,950 rows and 11 columns, representing customer purchase details such as product, quantity, price, order date, and purchase location. The dataset also includes derived fields like month, sales amount, city, and hour extracted from timestamps.

Columns in the Dataset

Order ID – Unique identifier for each order

Product – Name of the product purchased

Quantity Ordered – Number of units purchased

Price Each – Price of a single unit

Order Date – Date and time when the order was made

Purchase Address – Full customer delivery address

Month – Extracted month of purchase

Sales – Total sales value (Quantity × Price)
## Features analyzed include:
Features analyzed include product details, order quantities, pricing information, total sales, timestamp-based features (month and hour), and location-based features such as city and purchase address.
## Tools and Libraries Used

Python, Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualizations
Label Encoding for categorical data transformation, StandardScaler for numerical scaling.
## PROGRAM AND OUTPUT
```
from google.colab import files
uploaded= files.upload()

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensuring proper style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12,6)


# 2. Load Dataset

df = pd.read_csv("Sales Data.csv")

print("Initial Shape:", df.shape)
df.head()
```
<img width="1249" height="441" alt="image" src="https://github.com/user-attachments/assets/b86700d3-4247-4585-90c2-f13d4eaf6260" />  

## Cleaning:

```

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=["Unnamed: 0"])    # Remove unwanted index column
df = df.dropna()
print("After NA Removal:", df.shape)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
# Total Sales
df["Sales"] = df["Quantity Ordered"] * df["Price Each"]

# Extract month & hour
df["Month"] = df["Order Date"].dt.month
df["Hour"] = df["Order Date"].dt.hour
```
<img width="638" height="67" alt="image" src="https://github.com/user-attachments/assets/03337d4a-1e1d-4650-9564-b52f361b4681" />   


## EDA:
```
print(df.describe())
print(df.info())
```
<img width="784" height="429" alt="image" src="https://github.com/user-attachments/assets/0650ad99-834b-4e41-be9b-db35470c98e2" />
<img width="779" height="601" alt="image" src="https://github.com/user-attachments/assets/92d8aa6f-488e-40a2-b90d-e7dd8032add0" />

## FEATURE ENCODING:   

```
from sklearn.preprocessing import LabelEncoder

label_cols = ["City", "Product", "Purchase Address"]
encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

from sklearn.preprocessing import LabelEncoder

label_cols = ["City", "Product", "Purchase Address"]
encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])
```
<img width="624" height="370" alt="image" src="https://github.com/user-attachments/assets/d43f3a0d-cfb7-412e-9a2d-0685aaab4855" />   

## FEATURE SELECTION:   

```
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
<img width="1027" height="807" alt="image" src="https://github.com/user-attachments/assets/3f86ab5b-6f6e-4156-b423-edf699b22b71" />     

## DATA VISUALIZATION (Matplotlib + Seaborn):   

## Sales by Month (Matplotlib)   

```
plt.plot(monthly_sales.index, monthly_sales.values)
plt.title("Total Sales per Month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```
<img width="1011" height="556" alt="image" src="https://github.com/user-attachments/assets/b06a3b39-ae97-4cb3-a456-fc0d79c957f8" />   

## Most Selling Products (Seaborn):   

```
product_sales = df.groupby("Product")["Quantity Ordered"].sum().sort_values(ascending=False)

sns.barplot(x=product_sales.values, y=product_sales.index)
plt.title("Most Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Products")
plt.show()
```
<img width="1020" height="556" alt="image" src="https://github.com/user-attachments/assets/0a350902-f4cf-4263-947c-39dcee2f9ba5" />   

## Sales by City :   

```
city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False)

sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.show()
```
<img width="996" height="558" alt="image" src="https://github.com/user-attachments/assets/fa5b09a5-05c9-4aac-8fa8-0e5a68d3ec8d" />   
 
## Best Time to Sell (Hour-wise Sales):   

```
hourly_sales = df.groupby("Hour")["Sales"].sum()

plt.plot(hourly_sales.index, hourly_sales.values)
plt.title("Sales by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```
<img width="1010" height="556" alt="image" src="https://github.com/user-attachments/assets/1e742775-b5b8-419d-ae74-a958e4514d92" />














