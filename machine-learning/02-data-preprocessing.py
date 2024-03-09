from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def minmax_data():
    """
    The English translation of 归一化 is "normalization"
    There are many different ways to normalize data, but some common methods include:
    Min-max normalization: This method scales the data so that the minimum value is 0 and the maximum value is 1.
    Z-score normalization: This method subtracts the mean from each data point and then divides by the standard deviation.
    Decimal scaling: This method moves the decimal point of the data so that all values are integers.
    The best normalization method to use depends on the specific data set and the machine learning task being performed.

    Here are some examples of how normalization is used in practice:

    In image processing, normalization is used to improve the contrast of images.
    In natural language processing, normalization is used to remove stop words and stem words.
    In machine learning, normalization is used to prepare data for training and testing models.
    :return:
    """
    data = pd.read_csv("milag-data.csv")
    print(f'data: \n {data.head()}')

    transformer = MinMaxScaler()
    data_new = transformer.fit_transform(data)
    print(f'transformed data: \n {data_new[:5]}')
    print(f'the feature names: \n {transformer.get_feature_names_out()}')

    return None


# use StandardScaler to normalize the data
def standard_data():
    data = pd.read_csv("milag-data.csv")
    print(f'data: \n {data.head()}')

    transformer = StandardScaler()
    data_new = transformer.fit_transform(data)
    print(f'transformed data: \n {data_new[:5]}')
    print(f'the feature names: \n {transformer.get_feature_names_out()}')

    return None


def variance_data():
    """
    特征降维 in Chinese translates to "dimensionality reduction" in English. It is a process of reducing the number of features in a dataset while retaining as much information as possible. This can be useful for a variety of reasons, such as:

    Improving the performance of machine learning models. By reducing the number of features, models can be trained more quickly and efficiently.
    Making data more interpretable. By reducing the number of features, it can be easier to understand the relationships between different variables.
    Reducing the cost of data storage and processing. By reducing the number of features, less storage space is required and data can be processed more quickly.
    There are many different dimensionality reduction techniques available, each with its own advantages and disadvantages. Some of the most common techniques include:

    Principal component analysis (PCA): This technique identifies the directions of maximum variance in the data and projects the data onto these directions.
    Linear discriminant analysis (LDA): This technique identifies the directions that best discriminate between different classes of data.
    T-distributed stochastic neighbor embedding (t-SNE): This technique is a non-linear dimensionality reduction technique that can be used to visualize high-dimensional data.
    :return:
    """
    data = pd.read_csv("factor_returns.csv")
    print(f'data: \n {data.head()}')

    data = data.iloc[:, 1:-2]
    print(f'data: \n {data.head()}')

    transformer = VarianceThreshold(threshold=10)
    data_new = transformer.fit_transform(data)
    print(f'transformed data: \n {data_new[:5]}')
    print(f'the data shape of the transformed data: \n {data_new.shape}')

    # check the data correlation between all the feature names
    for feature_name_i in transformer.get_feature_names_out():
        for feature_name_j in transformer.get_feature_names_out():
            if feature_name_i != feature_name_j:
                result = correlation(data[feature_name_i], data[feature_name_j], x_name=feature_name_i,
                                     y_name=feature_name_j)
                if result[0] > 0.6:
                    plt.figure(figsize=(10, 10), dpi=60)
                    plt.scatter(data[feature_name_i], data[feature_name_j])
                    plt.xlabel(feature_name_i)
                    plt.ylabel(feature_name_j)
                    plt.title(f'The correlation - {feature_name_i} vs {feature_name_j}')
                    plt.show()

    return None


# define a method to calculate the correlation between two variables
def correlation(x, y, x_name=None, y_name=None):
    """
    This method is used to calculate the correlation between two variables
    :param x: the first variable
    :param y: the second variable
    :return: the correlation between the two variables
    """
    # calculate the correlation between the two variables
    correlation = pearsonr(x, y)

    if x_name is None or y_name is None:
        print(f'The correlation between the two variables is: {correlation}')
    else:
        print(f'The correlation between [{x_name}] and [{y_name}] is: {correlation}')

    return correlation

# define a method for PCA
def pca_data():
    """
    Principal component analysis (PCA) is a technique for reducing the dimensionality of data. It works by identifying the directions of maximum variance in the data and projecting the data onto these directions. This can be useful for a variety of reasons, such as:

    Reducing the number of features in a dataset. By identifying the directions of maximum variance, PCA can be used to identify a smaller number of features that capture most of the information in the data.
    Visualizing high-dimensional data. By projecting the data onto a lower-dimensional space, PCA can be used to visualize high-dimensional data in two or three dimensions.
    Identifying patterns in data. By identifying the directions of maximum variance, PCA can be used to identify patterns in the data that may not be apparent in the original feature space.
    There are many different techniques for performing PCA, each with its own advantages and disadvantages. Some of the most common techniques include:

    Singular value decomposition (SVD): This technique decomposes the data matrix into three matrices: U, Σ, and V. The columns of U and V are the left and right singular vectors, and Σ is a diagonal matrix of singular values.
    Eigenvalue decomposition: This technique decomposes the covariance matrix of the data into eigenvectors and eigenvalues. The eigenvectors are the directions of maximum variance, and the eigenvalues are the variances in these directions.
    Randomized PCA: This technique is a fast approximation of PCA that uses random projections to estimate the principal components of the data.
    :return:
    """
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    transformer = PCA(n_components=0.95)
    data_new = transformer.fit_transform(data)
    print(f'transformed data: \n {data_new}')
    print(f'the data type of the transformed data: \n {data_new.shape}')

    return None


if __name__ == "__main__":
    # minmax_data()
    # standard_data()
    # variance_data()
    pca_data()