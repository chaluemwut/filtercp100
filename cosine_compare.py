from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle, random, time
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

repeating_time = 10

clf = RandomForestClassifier()

def cosine_compare(d1, x_training_data, y_training_data):
    cos_lst = cosine_similarity(d1, x_training_data)
    idx = cos_lst.argmax(axis=1)
    return np.array(y_training_data)[idx]

def ml_prediction(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.feature_list) for x in x_train]
    x_test = [np.array(x.feature_list) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    # y_pred = cosine_compare(x_test, x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    return f1

def ml_word_prediction(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.feature_and_word_list) for x in x_train]
    x_test = [np.array(x.feature_and_word_list) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    # y_pred = cosine_compare(x_test, x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    return f1

def ml_cos_prediction(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.feature_list) for x in x_train]
    x_test = [np.array(x.feature_list) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    y_pred = cosine_compare(x_test, x_train, y_train)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    return f1

def ml_cos_word_prediction(x_train, x_test, y_train, y_test):
    x_train = [np.array(x.feature_and_word_list) for x in x_train]
    x_test = [np.array(x.feature_and_word_list) for x in x_test]
    y_train = [np.array(x) for x in y_train]
    y_test = [np.array(x) for x in y_test]

    y_pred = cosine_compare(x_test, x_train, y_train)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)

    f1 = f1_score(y_test, y_pred)
    return f1

def process():
    mapping_lst = pickle.load(open('data/data/data4000.data', 'rb'))
    x = []
    y = []
    for mapping in mapping_lst:
        x.append(mapping)
        y.append(mapping.prediction_result)

    f1_ml_lst = []
    f1_ml_word_lst = []
    f1_ml_cos_lst = []
    f1_ml_word_cos_lst = []
    for i in range(0, 10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randrange(1000))

        f1_ml = ml_prediction(x_train, x_test, y_train, y_test)
        f1_ml_word = ml_word_prediction(x_train, x_test, y_train, y_test)
        f1_ml_cos = ml_cos_prediction(x_train, x_test, y_train, y_test)
        f1_ml_word_cos = ml_cos_word_prediction(x_train, x_test, y_train, y_test)

        f1_ml_lst.append(f1_ml)
        f1_ml_word_lst.append(f1_ml_word)
        f1_ml_cos_lst.append(f1_ml_cos)
        f1_ml_word_cos_lst.append(f1_ml_word_cos)

    plt.boxplot([f1_ml_lst, f1_ml_word_lst, f1_ml_cos_lst, f1_ml_word_cos_lst])
    plt.show()

if __name__ == '__main__':
    process()