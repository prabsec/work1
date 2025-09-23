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


# In[7]:


df = load_data()
print(df.head())


# In[8]:


# Train & Evaluate
# ----------------------------
def evaluate_models():
    from pprint import pprint
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor

    # Load
    df = load_data()
    X = df.drop(columns=['MEDV']).values
    y = df['MEDV'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipelines
    pipe_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression())
    ])
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge())
    ])
    pipe_rf = Pipeline([
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # EXACTLY 3 HYPERPARAMETERS EACH
    param_grid_lr = {
        "model__fit_intercept": [True, False],
        "model__copy_X": [True, False],
        "model__positive": [False, True],
    }

    param_grid_ridge = {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
        "model__solver": ["auto", "svd", "cholesky", "lsqr"],
        "model__fit_intercept": [True, False],
    }

    param_grid_rf = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
    }

    # Grid searches
    searches = {
        "LinearRegression": GridSearchCV(
            estimator=pipe_lr,
            param_grid=param_grid_lr,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        ),
        "Ridge": GridSearchCV(
            estimator=pipe_ridge,
            param_grid=param_grid_ridge,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        ),
        "RandomForest": GridSearchCV(
            estimator=pipe_rf,
            param_grid=param_grid_rf,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        ),
    }

    # Fit & evaluate
    results = []
    best_params_out = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))
        best_params_out[name] = search.best_params_

    # Print results
    print("Model Performance on Boston Housing Dataset (with Hyperparameter Tuning)")
    print("{:<15} {:<15} {:<15}".format("Model", "MSE", "R²"))
    for name, mse, r2 in results:
        print("{:<15} {:<15.3f} {:<15.3f}".format(name, mse, r2))

    print("\nBest hyperparameters found (via 5-fold CV on train set):")
    pprint(best_params_out)


if __name__ == "__main__":
    # quick peek
    df = load_data()
    print(df.head())
    # run tuning + evaluation
    evaluate_models()


# In[ ]:





# In[ ]:




