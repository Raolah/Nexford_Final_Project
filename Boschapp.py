#!/usr/bin/env python
# coding: utf-8

# PRICE OPTIMIZATION FOR BOSCH

# DATA CLEANING AND PREPROCESSING

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('retail_price.csv')

# Display the first few rows of the dataframe
print(df.head())

# Data Preprocessing

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['freight_price'] = imputer.fit_transform(df[['freight_price']])
df['product_weight_g'] = imputer.fit_transform(df[['product_weight_g']])

# Encode categorical variables
categorical_features = ['product_category_name']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Normalize numerical features
numerical_features = ['qty', 'total_price', 'freight_price', 'product_name_lenght',
                      'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score',
                      'customers', 'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume',
                      'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

print("Data cleaning and preprocessing completed. The cleaned data is saved to 'cleaned_preprocessed.csv'.")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X = df.drop(['unit_price'], axis=1)
y = df['unit_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing pipeline to the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Save the cleaned and preprocessed dataset to a new CSV file
df.to_csv('cleaned_preprocessed.csv', index=False)

print("Data preprocessing completed successfully.")


# PERFORM EXPLORATOYR DATA ANALYSIS

# In[2]:


# Perform exploratory data analysis (EDA)
# Summary statistics of the dataset
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)


# DATA VISUALIZATION

# DEMAND QUANTITY DISTRIBUTION

# In[3]:


# Distribution of demand (qty)
plt.figure(figsize=(10, 6))
sns.histplot(df['qty'], kde=True, bins=30)
plt.title('Distribution of Demand (Quantity)')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()


# UNIT PRICE DISTRIBUTION

# In[4]:


# Distribution of unit price
plt.figure(figsize=(10, 6))
sns.histplot(df['unit_price'], kde=True, bins=30)
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.show()


# DEMAND VS COMPETITOR PRICES

# In[5]:


# Scatter plot of demand vs. competitor prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x='comp_1', y='qty', data=df, label='Competitor 1')
sns.scatterplot(x='comp_2', y='qty', data=df, label='Competitor 2')
sns.scatterplot(x='comp_3', y='qty', data=df, label='Competitor 3')
plt.title('Demand vs. Competitor Prices')
plt.xlabel('Competitor Prices')
plt.ylabel('Demand (Quantity)')
plt.legend()
plt.show()


# DEMAND VS UNIT PRICE

# In[6]:


# Scatter plot of demand vs. unit price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='unit_price', y='qty', data=df)
plt.title('Demand vs. Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Demand (Quantity)')
plt.show()


# DEMAND BY PRODUCT CATEGORY

# In[7]:


# Box plot of demand by product category
plt.figure(figsize=(12, 8))
sns.boxplot(x='product_category_name', y='qty', data=df)
plt.title('Demand by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Demand (Quantity)')
plt.xticks(rotation=90)
plt.show()


# DEMAND ELASTICITY ANALYSIS

# In[9]:


# Demand Elasticity Analysis
def demand_elasticity_analysis(df):
    # Calculate price elasticity of demand
    df['price_change'] = df['unit_price'].pct_change()
    df['qty_change'] = df['qty'].pct_change()
    
    elasticity = df['qty_change'].corr(df['price_change'])
    
    print(f"Price Elasticity of Demand: {elasticity}")

demand_elasticity_analysis(df)


# COMPETITOR ANALYSIS

# In[10]:


# Competitor Analysis
def competitor_analysis(df):
    # Compare Bosch’s prices with competitors
    competitors = ['comp_1', 'comp_2', 'comp_3']
    
    for comp in competitors:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='month_year', y=comp, label=comp)
        sns.lineplot(data=df, x='month_year', y='unit_price', label='Bosch')
        plt.title(f'Price Comparison with {comp}')
        plt.xlabel('Month-Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

competitor_analysis(df)


# FEATURE ENGINEERING AND MODEL DEVELOPMENT

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('retail_price.csv')

# Data Preprocessing

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['freight_price'] = imputer.fit_transform(df[['freight_price']])
df['product_weight_g'] = imputer.fit_transform(df[['product_weight_g']])

# Encode categorical variables
categorical_features = ['product_category_name']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Normalize numerical features
numerical_features = ['qty', 'total_price', 'freight_price', 'product_name_lenght',
                      'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score',
                      'customers', 'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume',
                      'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X = df.drop(['unit_price'], axis=1)
y = df['unit_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing pipeline to the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Feature Engineering

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_preprocessed)
X_test_poly = poly.transform(X_test_preprocessed)

# Model Training

# Define the model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_poly, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Model Development using XGBoost
xgb_model = XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_preprocessed, y_train)

y_pred_xgb = xgb_model.predict(X_test_preprocessed)

print("XGBoost Model Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_xgb)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_xgb)}")
print(f"R-squared: {r2_score(y_test, y_pred_xgb)}")

# Optimization using XGBoost model
optimal_prices = xgb_model.predict(preprocessor.transform(X))
df['optimal_price'] = optimal_prices

# Validation and Testing using XGBoost model
y_pred_validation = xgb_model.predict(preprocessor.transform(X))

print("Validation and Testing Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred_validation)}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_validation)}")
print(f"R-squared: {r2_score(y, y_pred_validation)}")

# Model Evaluation

# Predict on the test set
y_pred = best_model.predict(X_test_poly)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Feature Importance
importances = best_model.feature_importances_
feature_names = poly.get_feature_names_out(input_features=numerical_features + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)))
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print(feature_importances.head(10))


# SAVE AND TRAIN THE MODEL TO CREATE A FLASK API

# In[12]:


from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




