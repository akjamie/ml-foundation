from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# use Sklearn iris dataset to demonstrate the classification algorithm of KNN
def knn_iris():
    """
    This method is used to demonstrate the classification algorithm of KNN
    :return:
    """
    # get dataset
    iris = load_iris()

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # extract features
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # KNN algorithm estimator
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train);

    # predict the test data
    y_predict = estimator.predict(x_test)
    print(f'The predicted result is: \n {y_predict}')
    print(f'The actual result is: \n {y_test}')
    print(f'Compare the predicted result with the actual result: \n {y_predict == y_test}')
    print(f'The accuracy of the KNN algorithm is: \n {estimator.score(x_test, y_test)}')


# add cross validation and grid search to the iris_knn method
def iris_knn_with_gridsearch_cv():
    """
    This method is used to add cross validation and grid search to the iris_knn method and apply the best parameters to
    the KNN algorithm
    :return:
    """
    # get dataset
    iris = load_iris()

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # extract features
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # KNN algorithm estimator
    estimator = KNeighborsClassifier()

    # define the parameters to be searched
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}

    # grid search
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # predict the test data
    y_predict = estimator.predict(x_test)
    print(f'The predicted result is: \n {y_predict}')
    print(f'The actual result is: \n {y_test}')
    print(f'Compare the predicted result with the actual result: \n {y_predict == y_test}')
    print(f'The accuracy of the KNN algorithm is: \n {estimator.score(x_test, y_test)}')
    print(f'The best parameters are: \n {estimator.best_params_}')
    print(f'The best score is: \n {estimator.best_score_}')
    print(f'The best estimator is: \n {estimator.best_estimator_}')
    print(f'The cross validation results are: \n {estimator.cv_results_}')


# define method to use naive bayes
def news_naive_bayes():
    """
    This method is used to demonstrate the naive bayes algorithm
    :return:
    """
    # Fetch the 20 newsgroups dataset
    news = fetch_20newsgroups(subset='all')

    # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # extract features
    transformer = TfidfVectorizer()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # naive bayes algorithm estimator
    estimator = MultinomialNB(alpha=0.001)
    estimator.fit(x_train, y_train)

    # predict the test data
    y_predict = estimator.predict(x_test)
    print(f'The predicted result is: \n {y_predict}')
    print(f'The actual result is: \n {y_test}')
    print(f'Compare the predicted result with the actual result: \n {y_predict == y_test}')

    print(f'The accuracy of the naive bayes algorithm is: \n {estimator.score(x_test, y_test)}')


def decision_iris():
    """
    To user decision tree to classify the iris dataset.
    :return:
    """
    # dataset
    iris_data = load_iris()

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, random_state=22)

    # estimator for decision tree
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=22)
    estimator.fit(x_train, y_train)

    # model training
    # predict the test data
    y_predict = estimator.predict(x_test)
    print(f'The predicted result is: \n {y_predict}')
    print(f'The actual result is: \n {y_test}')
    print(f'Compare the predicted result with the actual result: \n {y_predict == y_test}')

    print(f'The accuracy of the naive bayes algorithm is: \n {estimator.score(x_test, y_test)}')

    # export the decision tree
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris_data.feature_names)


if __name__ == "__main__":
    # knn_iris()
    print('\n--------------------\n')
    # iris_knn_with_gridsearch_cv()
    # news_naive_bayes()
    decision_iris()
