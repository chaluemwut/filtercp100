import pickle, random, time, logging, sys
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp import CRFWordSegment

from sklearn.metrics import f1_score
import numpy as np

class MainCompare():

    crf = CRFWordSegment()

    time_train_ml = []
    time_predict_ml = []

    time_train_ml_word = []
    time_predict_ml_word = []

    time_train_topic = []
    time_predict_topic = []

    time_train_text = []
    time_predict_text = []

    repeating_time = 10

    log = logging.getLogger('resize')
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = logging.FileHandler("resize.log")
    fh.setFormatter(format)
    log.addHandler(fh)

    def ml_prediction(self, x_train, x_test, y_train, y_test):
        clf = RandomForestClassifier()
        x_train = [np.array(x.feature_list) for x in x_train]
        x_test = [np.array(x.feature_list) for x in x_test]
        y_train = [np.array(x) for x in y_train]
        y_test = [np.array(x) for x in y_test]

        start_time = time.time()
        clf.fit(x_train, y_train)
        total_time = time.time() - start_time
        self.time_train_ml.append(total_time / len(y_train))

        start_time = time.time()
        y_pred = clf.predict(x_test)
        total_time = time.time() - start_time
        self.time_predict_ml.append(total_time)

        f1 = f1_score(y_test, y_pred)
        return f1

    def ml_word_prediction(self, x_train, x_test, y_train, y_test):
        clf = RandomForestClassifier()
        x_train = [np.array(x.feature_and_word_list) for x in x_train]
        x_test = [np.array(x.feature_and_word_list) for x in x_test]
        y_train = [np.array(x) for x in y_train]
        y_test = [np.array(x) for x in y_test]

        start_time = time.time()
        clf.fit(x_train, y_train)
        total_time = time.time() - start_time
        self.time_train_ml_word.append(total_time / len(y_train))

        start_time = time.time()
        y_pred = clf.predict(x_test)
        total_time = time.time() - start_time
        self.time_predict_ml_word.append(total_time / len(y_pred))

        f1 = f1_score(y_test, y_pred)
        return f1

    def to_message_lst(self, msg_obj):
        msg_seg = self.crf.crfpp(msg_obj.message)
        msg_data = ' '.join(msg_seg)
        return msg_data

    def topic_detection(self, x_train, x_test, y_train, y_test):
        tfidf_vectorizer = TfidfVectorizer()

        x_train_inner, x_test_inner, y_train_inner, y_test_inner = train_test_split(x_train, y_train,
                                                                                    test_size=0.2,
                                                                                    random_state=random.randrange(1000))

        x_train_msg_inner = []
        for x_msg in x_train_inner:
            data_lst = self.crf.crfpp(x_msg.message)
            data_msg = ' '.join(data_lst)
            x_train_msg_inner.append(data_msg)

        cosin_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        f1_lst = []
        start_time = time.time()
        for cosin in cosin_lst:
            self.log.info('****** {} *******'.format(cosin))
            y_pred_lst = []
            for x_inner in x_test_inner:
                test_message = self.to_message_lst(x_inner)
                x_train_msg_inner.append(test_message)
                tfidf_matrix = tfidf_vectorizer.fit_transform(x_train_msg_inner)
                cos_lst = np.sort(cosine_similarity(tfidf_matrix[-1:], tfidf_matrix))[0]
                sim_max = cos_lst[len(cos_lst) - 2]
                if sim_max > cosin:
                    y_pred_lst.append(1)
                else:
                    y_pred_lst.append(0)
                del x_train_msg_inner[-1]
            f1 = f1_score(y_test_inner, y_pred_lst)
            f1_lst.append(f1)

        f1_lst = np.array(f1_lst)
        f1_max_idx = f1_lst.argmax()
        cosin_max = cosin_lst[f1_max_idx]

        total_time = time.time() - start_time
        self.time_train_topic.append(total_time / len(y_train_inner))

        x_test_corpus = []
        per_y_pred = []
        start_time = time.time()
        for x_data in x_test:
            data_seg = self.crf.crfpp(x_data.message)
            data = ' '.join(data_seg)
            x_test_corpus.append(data)

        for x in x_test:
            test_message = self.to_message_lst(x)
            x_test_corpus.append(test_message)
            tfidf_test = tfidf_vectorizer.fit_transform(x_test_corpus)
            cos_lst = np.sort(cosine_similarity(tfidf_test[-1:], tfidf_test))[0]
            sim_max = cos_lst[len(cos_lst) - 2]
            if sim_max > cosin_max:
                per_y_pred.append(1)
            else:
                per_y_pred.append(0)

        total_time = time.time() - start_time
        self.time_predict_topic.append(total_time / len(y_test))

        f1 = f1_score(y_true=y_test, y_pred=per_y_pred)
        return f1

    def text_mining(self, x_train, x_test, y_train, y_test):
        x_train_msg = []
        x_test_msg = []
        crf = CRFWordSegment()
        for x_msg in x_train:
            data_lst = crf.crfpp(x_msg.message)
            data_msg = ' '.join(data_lst)
            x_train_msg.append(data_msg)

        for x_msg in x_test:
            data_lst = crf.crfpp(x_msg.message)
            data_msg = ' '.join(data_lst)
            x_test_msg.append(data_msg)

        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', RandomForestClassifier())])
        start_time = time.time()
        text_clf = text_clf.fit(x_train_msg, y_train)
        total_time = time.time() - start_time
        self.time_train_text.append(total_time / len(y_train))

        start_time = time.time()
        y_pred = text_clf.predict(x_test_msg)
        total_time = time.time() - start_time
        self.time_predict_text.append(total_time / len(y_pred))

        f1 = f1_score(y_test, y_pred)
        return f1

    def print_all_result(self, all_result):
        print('********** performance result')
        for ml, ml_word, topic, perf_text in zip(all_result['perf_ml'], all_result['perf_ml_word'],
                                                 all_result['perf_topic'], all_result['perf_text']):
            self.log.info('{},{},{},{}'.format(ml, ml_word, topic, perf_text))

        print('********** training time')
        for t_ml, t_ml_word, t_topic, t_text in zip(all_result['time_train_ml'], all_result['time_train_ml_word'],
                                                    all_result['time_train_topic'], all_result['time_train_text']):
            self.log.info('{},{},{},{}'.format(t_ml, t_ml_word, t_topic, t_text))

        for p_ml, p_ml_word, p_topic, p_text in zip(all_result['time_predict_ml'], all_result['time_predict_ml_word'],
                                                    all_result['time_predict_topic'], all_result['time_predict_text']):
            self.log.info('{},{},{},{}'.format(p_ml, p_ml_word, p_topic, p_text))

    def main_process(self, test_size):
        mapping_lst = pickle.load(open('data/data/data4000.data', 'rb'))
        x = []
        y = []
        for mapping in mapping_lst:
            x.append(mapping)
            y.append(mapping.prediction_result)

        ml_lst = []
        ml_word_lst = []
        topic_lst = []
        text_lst = []
        for i in range(0, self.repeating_time):
            self.log.info('****** start loop {} '.format(i))
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                                random_state=random.randrange(1000))

            ml_result = self.ml_prediction(x_train, x_test, y_train, y_test)
            ml_word_result = self.ml_word_prediction(x_train, x_test, y_train, y_test)
            topic_result = self.topic_detection(x_train, x_test, y_train, y_test)
            text_result = self.text_mining(x_train, x_test, y_train, y_test)

            ml_lst.append(ml_result)
            ml_word_lst.append(ml_word_result)
            topic_lst.append(topic_result)
            text_lst.append(text_result)
            self.log.info('[ml : {}, text : {}, ml word : {}, topic : {}]'.format(ml_result, text_result,
                                                                             ml_word_result, topic_result))
            self.log.info('****** end loop {} '.format(i))

        all_result = {}
        all_result['perf_ml'] = ml_lst
        all_result['perf_ml_word'] = ml_word_lst
        all_result['perf_topic'] = topic_lst
        all_result['perf_text'] = text_lst

        all_result['time_train_ml'] = self.time_train_ml
        all_result['time_predict_ml'] = self.time_predict_ml

        all_result['time_train_ml_word'] = self.time_train_ml_word
        all_result['time_predict_ml_word'] = self.time_predict_ml_word

        all_result['time_train_topic'] = self.time_train_topic
        all_result['time_predict_topic'] = self.time_predict_topic

        all_result['time_train_text'] = self.time_train_text
        all_result['time_predict_text'] = self.time_predict_text

        self.print_all_result(all_result)

        return all_result

