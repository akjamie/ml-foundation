from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib


# define method to do fillin the missing data
def fill_missing_data(df):
    print(f'Before the fillna\n: {df.isna().sum()}')
    for col in df.columns:
        if df[col].dtype != 'object':
            # df[col].fillna(df[col].mean(), inplace=True)
            df.fillna({col: df[col].mean()}, inplace=True)

    print(f'After the fillna\n: {df.isna().sum()} \n')


# define method to do linear regression(Normal Equation) for boston price prediction
def linear_regression_normal_equation():
    # load dataset
    boston = pd.read_csv("HousingData.csv")
    print(f'Data shape: {boston.shape}')
    fill_missing_data(boston)

    x = boston.iloc[:, :-1]
    y = boston['MEDV']

    # split dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # normalize data
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # linear regression
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # model evaluation
    print("linear regression model[normal_equation] coef: ", estimator.coef_)
    print("linear regression model[normal_equation] intercept: ", estimator.intercept_)
    print("linear regression model[normal_equation] score: ", estimator.score(x_test, y_test))

    y_predict = estimator.predict(x_test)
    print(f'Predicted price: {y_predict}')
    error = mean_squared_error(y_test, y_predict)
    print(f'Error for normal equation : {error}')


# define method to do linear regression(Gradient descent) for boston price prediction
def linear_regression_gradient_descent():
    # load dataset
    boston = pd.read_csv("HousingData.csv")
    fill_missing_data(boston)
    x = boston.iloc[:, :-1]
    y = boston['MEDV']

    # split dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # normalize data
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # linear regression
    # estimator = SGDRegressor()
    estimator = SGDRegressor(penalty='l1', learning_rate='constant', eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)

    # model evaluation
    print("linear regression model[gradient_descent] coef: ", estimator.coef_)
    print("linear regression model[gradient_descent] intercept: ", estimator.intercept_)
    print("linear regression model[gradient_descent] score: ", estimator.score(x_test, y_test))

    y_predict = estimator.predict(x_test)
    print(f'Predicted price: {y_predict}')
    error = mean_squared_error(y_test, y_predict)
    print(f'Error for normal equation : {error}')


# define method to do linear regression(Ridge) for boston price prediction
def linear_regression_ridge():
    # load dataset
    boston = pd.read_csv("HousingData.csv")
    fill_missing_data(boston)
    x = boston.iloc[:, :-1]
    y = boston['MEDV']

    # split dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # normalize data
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # linear regression
    # estimator = SGDRegressor(penalty='l2')
    # estimator = Ridge(alpha=0.05, max_iter=10000)
    # estimator.fit(x_train, y_train)

    # joblib.dump(estimator, 'ridge.pkl')
    estimator = joblib.load('ridge.pkl')

    # model evaluation
    print("linear regression model[ridge] coef: ", estimator.coef_)
    print("linear regression model[ridge] intercept: ", estimator.intercept_)
    print("linear regression model[ridge] score: ", estimator.score(x_test, y_test))

    y_predict = estimator.predict(x_test)
    print(f'Predicted price: {y_predict}')
    error = mean_squared_error(y_test, y_predict)
    print(f'Error for normal equation : {error}')


if __name__ == "__main__":
    #linear_regression_normal_equation()
    #linear_regression_gradient_descent()
    linear_regression_ridge()
