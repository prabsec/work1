#!/usr/bin/env python
# coding: utf-8

# In[1]:


def load_data():
    import pandas as pd
    import numpy as np
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # now we split the data into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # These are feature names
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    # create a data frame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target # here MEDV is our target variable

    return df


# In[11]:


df = load_data()
print(df.head())


# In[13]:


# Train & Evaluate
# ----------------------------
def evaluate_models():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor

    df = load_data()
    X = df.drop(columns=['MEDV']).values
    y = df['MEDV'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models (Lasso removed)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    }

    # Evaluate each model
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))

    # Print results in table
    print("Model Performance on Boston Housing Dataset")
    print("{:<15} {:<15} {:<15}".format("Model", "MSE", "R²"))
    for name, mse, r2 in results:
        print("{:<15} {:<15.3f} {:<15.3f}".format(name, mse, r2))


if __name__ == "__main__":
    evaluate_models()


# In[ ]:





# In[ ]:




