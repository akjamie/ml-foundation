from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import random


# define a method to return the iris dataset
def iris_dataset():
    # define a variable to store the iris dataset
    iris = load_iris()

    # print the iris dataset
    print(iris);

    # return the iris dataset
    return iris;


def dict_features_extraction():
    """
    This method is used to extract the features from the given dataset
    :return:
    """
    # define an 2 dimensional array to store the temperature data for 3 cities
    data = [{"city": "New-York", "temperature": 100, "humidity": 80, "wind": 10},
            {"city": "Chicago", "temperature": 30, "humidity": 60, "wind": 20},
            {"city": "San-Francisco", "temperature": 28, "humidity": 70, "wind": 15}]

    # instantiate transformer
    transformer = DictVectorizer(sparse=False);

    # transform the data
    data_new = transformer.fit_transform(data);
    print(f'The transformed data is: \n {data_new}')
    print(f'The feature names are: \n {transformer.get_feature_names_out()}')

    # inverse transform the data
    print(f'The inverse transformed data is: \n {transformer.inverse_transform(data_new)}')

    return None;


def text_features_extraction():
    """
    This method is used to extract the features from the given text data
    :return:
    """
    # define an 2 dimensional array to store the temperature data for 3 cities
    data = ["Life sucks but you gonna love it.",
            "I deeply love this city where i was born.",
            "Your youthful appearance has been imprinted in my mind.",
            "I love the city more than you can imagine."]

    # instantiate transformer
    transformer = CountVectorizer(stop_words=['you', 'the']);

    # transform the data
    data_new = transformer.fit_transform(data);
    print(f'The transformed data is: \n {data_new}')
    print(f'The transformed data type is: \n {type(data_new)}')
    print(f'The transformed data is: \n {data_new.toarray()}')
    print(f'The feature names are: \n {transformer.get_feature_names_out()}')

    # inverse transform the data
    print(f'The inverse transformed data is: \n {transformer.inverse_transform(data_new)}')

    return None;


def text_features_extraction_chinese():
    """
    This method is used to extract the features from the given text data
    :return:
    """
    # define an 2 dimensional array to store the temperature data for 3 cities
    data = ["人生 如此 艰难，但 你 会 喜欢它。",
            "我 深深地 爱着 我 出生的 这个 城市。",
            "你年轻的外表已经印在我的脑海里。",
            "我爱这个城市，超出你的想象。"]

    # instantiate transformer
    transformer = CountVectorizer();

    # transform the data
    data_new = transformer.fit_transform(data);
    print(f'The transformed data is: \n {data_new}')
    print(f'The transformed data type is: \n {type(data_new)}')
    print(f'The transformed data is: \n {data_new.toarray()}')
    print(f'The feature names are: \n {transformer.get_feature_names_out()}')

    # inverse transform the data
    print(f'The inverse transformed data is: \n {transformer.inverse_transform(data_new)}')

    return None;


# define a method to extract features from Chinese characters with using jieba.
def text_features_extraction_chinese_jieba():
    data = ["轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。",
            "那河畔的金柳，是夕阳中的新娘；波光里的艳影，在我的心头荡漾。",
            "软泥上的青荇，油油的在水底招摇；在康河的柔波里，我甘心做一条水草！",
            "那榆荫下的一潭，不是清泉，是天上虹；揉碎在浮藻间，沉淀着彩虹似的梦。",
            "撑一支长篙，向青草更青处漫溯；满载一船星辉，在星辉斑斓里放歌。",
            "但我不能放歌，悄悄是别离的笙箫；夏虫也为我沉默，沉默是今晚的康桥！",
            "悄悄的我走了，正如我悄悄的来；我挥一挥衣袖，不带走一片云彩。"]
    print(f'The original data is: \n {data}')

    data_cut = []
    for sentence in data:
        words = cut_words(sentence)
        data_cut.append(words)

    print(f'The cut data is: \n {data_cut}')

    # instantiate transformer
    transformer = CountVectorizer();

    # transform the data
    data_new = transformer.fit_transform(data_cut);
    print(f'The transformed data is: \n {data_new}')
    print(f'The transformed data type is: \n {type(data_new)}')
    print(f'The transformed data is: \n {data_new.toarray()}')
    print(f'The feature names are: \n {transformer.get_feature_names_out()}')

    return None;

def text_features_extraction_chinese_jieba_tfidf():
    data = ["轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。",
            "那河畔的金柳，是夕阳中的新娘；波光里的艳影，在我的心头荡漾。",
            "软泥上的青荇，油油的在水底招摇；在康河的柔波里，我甘心做一条水草！",
            "那榆荫下的一潭，不是清泉，是天上虹；揉碎在浮藻间，沉淀着彩虹似的梦。",
            "撑一支长篙，向青草更青处漫溯；满载一船星辉，在星辉斑斓里放歌。",
            "但我不能放歌，悄悄是别离的笙箫；夏虫也为我沉默，沉默是今晚的康桥！",
            "悄悄的我走了，正如我悄悄的来；我挥一挥衣袖，不带走一片云彩。"]
    print(f'The original data is: \n {data}')

    data_cut = []
    for sentence in data:
        words = cut_words(sentence)
        data_cut.append(words)

    print(f'The cut data is: \n {data_cut}')

    # instantiate transformer
    transformer = TfidfVectorizer();

    # transform the data
    data_new = transformer.fit_transform(data_cut);
    print(f'The transformed data is: \n {data_new}')
    print(f'The transformed data type is: \n {type(data_new)}')
    print(f'The transformed data is: \n {data_new.toarray()}')
    print(f'The feature names are: \n {transformer.get_feature_names_out()}')

    # Get the feature names
    feature_names = transformer.get_feature_names_out()

    # Get the TF-IDF scores for each word
    tfidf_scores = data_new.toarray()[0]

    # Sort words by TF-IDF score in descending order
    sorted_words = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

    # Print the 10 words with highest TF-IDF scores
    for word, score in sorted_words[:5]:
        print(word, score)
    return None;

def cut_words(text):
    return " ".join(jieba.cut(text));

def prepare_data():
    # Define the data range and precision
    milarge_range = (5000, 80000)  # Range for milarge in thousands of kilometers
    liters_range = (0.0, 15.0)  # Range for liters (float)
    consumtime_range = (0.0, 3.0)  # Range for consumtime (float)
    num_records = 150  # Number of data points

    # Generate data
    data = []
    for _ in range(num_records):
        milarge = random.randint(*milarge_range)
        liters = random.uniform(*liters_range)
        consumtime = random.uniform(*consumtime_range)
        target = random.randint(1, 3)  # Target: 0 or 1
        data.append({"milarge": milarge, "liters": round(liters, 8), "consumtime": round(consumtime, 8), "target": target})

    # Print the generated data
    print(data)

    # write the data to csv file with header=['milage', 'liters', 'consumtime', 'target']
    with open('milag-data.csv', 'w') as f:
        f.write('milarge,liters,consumtime,target\n')
        # write the data to the file
        for record in data:
            f.write(f"{record['milarge']},{record['liters']},{record['consumtime']},{record['target']}\n")


# call the method iris_dataset to get the iris dataset in main method
if __name__ == "__main__":
    # define a variable to store the iris dataset
    # iris = iris_dataset();
    #
    # print(f'the description of the iris dataset is: \n {iris['DESCR']}')
    # print(f'the feature names of the iris dataset is: \n {iris['feature_names']}')
    # print(f'the target names of the iris dataset is: \n {iris['target_names']}')
    # print(f'feature values:{iris['data']} \nshape: {iris.data.shape}')
    # print(f'target values:{iris['target']} \nshape: {iris.target.shape}')
    #
    # # test split the iris dataset
    # x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=30)
    # print(f'feature values:{x_train} \nshape: {x_train.shape}')
    # print(f'feature values:{y_train} \nshape: {y_train.shape}')

    # dict_features_extraction();
    # text_features_extraction();
    # text_features_extraction_chinese()

    # print(cut_words("轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。"))
    # text_features_extraction_chinese_jieba()
    text_features_extraction_chinese_jieba_tfidf()
    # prepare_data()