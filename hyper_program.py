#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[22]:


df = load_data()
print(df.head())


# In[36]:


def evaluate_models(param_name, param_values):
    """
    Tune one RandomForest hyperparameter across 3 values and print train/test metrics.

    Example:
        evaluate_models("max_depth", [None, 5, 10])
        evaluate_models("n_estimators", [100, 200, 500])
        evaluate_models("max_features", ["auto", "sqrt", 0.8])
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    assert len(param_values) == 3, "Provide exactly 3 values for the hyperparameter."

    df = load_data()
    X = df.drop(columns=['MEDV']).values
    y = df['MEDV'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    for val in param_values:
        # Base model (you can change other defaults if you wish)
        model = RandomForestRegressor(
            n_estimators=200,  # default; will be overridden if param_name == "n_estimators"
            random_state=42,
            n_jobs=-1
        )
        # Override the chosen hyperparameter
        model.set_params(**{param_name: val})

        model.fit(X_train, y_train)

        # Train metrics
        y_pred_tr = model.predict(X_train)
        tr_mse = mean_squared_error(y_train, y_pred_tr)
        tr_r2 = r2_score(y_train, y_pred_tr)

        # Test metrics
        y_pred_te = model.predict(X_test)
        te_mse = mean_squared_error(y_test, y_pred_te)
        te_r2 = r2_score(y_test, y_pred_te)

        # Pretty value for printing (handle None)
        val_print = "None" if val is None else str(val)
        results.append((val_print, tr_mse, tr_r2, te_mse, te_r2))

    # Print table
    print(f"RandomForest (tuning '{param_name}') – Training vs Testing Performance")
    print("{:<15} {:>12} {:>12} {:>12} {:>12}".format(param_name, "Train MSE", "Train R²", "Test MSE", "Test R²"))
    for v, tr_mse, tr_r2, te_mse, te_r2 in results:
        print("{:<15} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(v, tr_mse, tr_r2, te_mse, te_r2))

if __name__ == "__main__":
    # EXAMPLES (uncomment one at a time)
    evaluate_models("max_depth", [None, 5, 10])
    # evaluate_models("n_estimators", [100, 200, 500])
    # evaluate_models("min_samples_split", [2, 5, 10]) 



# In[ ]:





# In[ ]:




