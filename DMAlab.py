#knn#####################################################################################################knn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. KNN with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Results for k=1")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 2. Find the best k (1 to 20)
best_k = 1
best_accuracy = 0

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\nBest k: {best_k} with Accuracy: {best_accuracy:.2f}")

# 3. Final model using best k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

print("\nFinal Model Evaluation with Best k")
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred))
print("Accuracy:", accuracy_score(y_test, final_pred))

import matplotlib.pyplot as plt
error_rates = []
# Compute error rate for k values from 1 to 20
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred_k = model.predict(X_test)
    error = 1 - accuracy_score(y_test, pred_k)
    error_rates.append(error)

# Plotting error rate vs k
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), error_rates, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=8)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.xticks(range(1, 21))
plt.show()

# Naive bayes################################################################################################ Naive bayes

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#arimax sarimax ###########################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Create synthetic seasonal data
np.random.seed(0)
n = 100
time = pd.date_range(start='2023-01-01', periods=n, freq='D')
data = np.sin(np.linspace(0, 20, n)) + np.random.normal(scale=0.5, size=n)
ts = pd.Series(data, index=time)

# 2. ADF Test on original series
adf_orig = adfuller(ts)
print("📊 ADF Test - Original Series")
print("ADF Statistic:", round(adf_orig[0], 4))
print("p-value:", round(adf_orig[1], 4))
print("-" * 40)

# 3. Apply seasonal differencing (weekly seasonality, lag=7)
ts_diff = ts.diff(7).dropna()

# 4. ADF Test after seasonal differencing
adf_seasonal = adfuller(ts_diff)
print("📊 ADF Test - After Seasonal Differencing (lag=7)")
print("ADF Statistic:", round(adf_seasonal[0], 4))
print("p-value:", round(adf_seasonal[1], 4))
print("-" * 40)

# 5. Plot Original & Differenced Series
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(ts, label='Original')
plt.title("Original Time Series")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ts_diff, label='Seasonal Differenced', color='orange')
plt.title("After Seasonal Differencing (lag=7)")
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Plot ACF & PACF for Differenced Series (tick-like spikes)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_acf(ts_diff, ax=plt.gca(), lags=30)
plt.title("Autocorrelation (ACF)")

plt.subplot(1, 2, 2)
plot_pacf(ts_diff, ax=plt.gca(), lags=30, method='ywm')
plt.title("Partial Autocorrelation (PACF)")

plt.tight_layout()
plt.show()
# decision tree#####################################################################################
# Import necessary libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
 
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier 
 
# Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target 
 
# Create and train the Random Forest model 
clf = RandomForestClassifier(n_estimators=20, random_state=0) 
clf = clf.fit(X, y) 
 
# Check the number of estimators (trees) in the forest 
len(clf.estimators_)  # Output: 20 
 
# Plotting a single decision tree from the forest 
plt.figure(figsize=(15,10)) 
tree.plot_tree(clf.estimators_[0], filled=True) 
 
# Plotting all trees in the forest 
plt.figure(figsize=(15,10)) 
for i in range(len(clf.estimators_)): 
    tree.plot_tree(clf.estimators_[i], filled=True) 
 
# Print text representation of trees 
for i in range(len(clf.estimators_)): 
    print(tree.export_text(clf.estimators_[i]))

# time series using fb data##########################################################################################

# Install required packages 
# pip install pystan 
# conda install -c conda-forge fbprophet 
 
# Import libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
from fbprophet import Prophet 
from fbprophet.diagnostics import cross_validation, performance_metrics 
from fbprophet.plot import plot_cross_validation_metric 
 
# Load data 
df = pd.read_csv('airline_passengers.csv') 
 
# Initial data exploration 
print(df.head()) 
print(df.tail()) 
 
# Plot original data 
df.plot() 
 
# Rename columns as required by Prophet 
df.columns = ['ds', 'y'] 
df['ds'] = pd.to_datetime(df['ds']) 
 
# Drop specific NaN row if needed 
df.drop(144, axis=0, inplace=True) 
 
# Drop any remaining missing values 
df.dropna(axis=0, inplace=True) 
 
# Initialize Prophet model 
model = Prophet() 
 
# Fit the model 
model.fit(df) 
 
# Create future dates for next 365 days 
future_dates = model.make_future_dataframe(periods=365) 
 
# Make predictions 
prediction = model.predict(future_dates) 
 
# Plot forecast 
model.plot(prediction) 
 
# Plot forecast components (trend, weekly & yearly seasonality) 
model.plot_components(prediction) 
 
# Cross-validation 
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days') 
 
# Performance metrics 
df_p = performance_metrics(df_cv) 
print(df_p.head()) 
 
# Plot cross-validation RMSE 
fig = plot_cross_validation_metric(df_cv, metric='rmse')
# k-means################################################################################################
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# Load dataset (2 features for simple visualization)
data = load_wine()
X = data.data[:, :2]

# Try k from 2 to 10
best_k = 2
best_score = -1

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    if score > best_score:
        best_score = score
        best_k = k

print(f"✅ Best k = {best_k} with Silhouette Score = {round(best_score, 3)}")

# ✅ Loop and plot only from k = 2 up to best_k
for k in range(2, best_k + 1):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_samples = silhouette_samples(X, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X)])
    y_lower = 0

    for i in range(k):
        ith_sil_vals = sil_samples[labels == i]
        ith_sil_vals.sort()
        size = ith_sil_vals.shape[0]
        y_upper = y_lower + size
        color = plt.cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper

    ax1.axvline(x=score, color='red', linestyle='--')
    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster")
    ax1.set_yticks([])

    # Cluster scatter plot
    colors = plt.cm.nipy_spectral(labels.astype(float) / k)
    ax2.scatter(X[:, 0], X[:, 1], c=colors, s=40, alpha=0.7)
    centers = model.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Centroids')
    ax2.set_title("Cluster Visualization")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()

    plt.suptitle(f"k={k} | Silhouette Score = {round(score, 3)}", fontsize=14)
    plt.tight_layout()
    plt.show()
# house price prediction##############################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/content/train.csv')  # replace with actual path
print("Dataset Shape:", data.shape)
print(data.head())

missing = data.isnull().sum()
missing = missing[missing > 0]
print("Missing Values:\n", missing)

data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
data.fillna(data.median(numeric_only=True), inplace=True)

plt.figure(figsize=(8, 6))
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('SalePrice')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(16, 10))
numeric_data = data.select_dtypes(include=np.number)
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

top_corr = corr['SalePrice'].sort_values(ascending=False).head(10)
print("Top correlated features:\n", top_corr)

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath']
data['Age'] = data['YrSold'] - data['YearBuilt']
data = pd.get_dummies(data, columns=['Neighborhood', 'HouseStyle'],
drop_first=True)

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalSF',
'TotalBath', 'Age']
X = data[features]
y = data['SalePrice']

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# 9. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Prediction
y_pred = model.predict(X_test)

# 11. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# Data transformation ##################################################################################################

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, 
RobustScaler 
from sklearn.metrics import accuracy_score, classification_report 
 
# Load Titanic dataset (assuming dataset is available as 'titanic.csv') 
df = pd.read_csv('titanic.csv') 
 
# Display first few rows 
print(df.head()) 
 
# Drop irrelevant columns 
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, 
inplace=True) 
 
# Handle missing values 
df['Age'].fillna(df['Age'].median(), inplace=True) 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) 
 
# Convert categorical to numerical 
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True) 
 
# Define features and target 
X = df.drop('Survived', axis=1) 
y = df['Survived'] 
 
# Split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
 
# Perform different scaling techniques 
 
# 1. Standard Scaling 
scaler_std = StandardScaler() 
X_train_std = scaler_std.fit_transform(X_train) 
X_test_std = scaler_std.transform(X_test) 
 
# Logistic Regression with Standard Scaled data 
model_std = LogisticRegression() 
model_std.fit(X_train_std, y_train) 
y_pred_std = model_std.predict(X_test_std) 
 
print("StandardScaler Results:") 
print("Accuracy:", accuracy_score(y_test, y_pred_std)) 
print(classification_report(y_test, y_pred_std)) 
 
# 2. MinMax Scaling 
scaler_mm = MinMaxScaler() 
X_train_mm = scaler_mm.fit_transform(X_train) 
X_test_mm = scaler_mm.transform(X_test) 
 
# Logistic Regression with MinMax Scaled data 
model_mm = LogisticRegression() 
model_mm.fit(X_train_mm, y_train) 
y_pred_mm = model_mm.predict(X_test_mm) 
 
print("\nMinMaxScaler Results:") 
print("Accuracy:", accuracy_score(y_test, y_pred_mm)) 
print(classification_report(y_test, y_pred_mm)) 
 
# 3. Robust Scaling 
scaler_rb = RobustScaler() 
X_train_rb = scaler_rb.fit_transform(X_train) 
X_test_rb = scaler_rb.transform(X_test) 
 
# Logistic Regression with Robust Scaled data 
model_rb = LogisticRegression() 
model_rb.fit(X_train_rb, y_train) 
y_pred_rb = model_rb.predict(X_test_rb) 
 
print("\nRobustScaler Results:") 
print("Accuracy:", accuracy_score(y_test, y_pred_rb)) 
print(classification_report(y_test, y_pred_rb)) 
