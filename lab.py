from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np

def cls_cos(d1, x_training_data, y_training_data):
    cos_lst = cosine_similarity(d1, x_training_data)
    idx = cos_lst.argmax(axis=1)
    return np.array(y_training_data)[idx]

if __name__ == '__main__':
    vectorizer = CountVectorizer()
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    corpus.append('this is a pen')
    tfidf = vectorizer.fit_transform(corpus).toarray()
    cos = cls_cos(tfidf[0:1], tfidf, [1,1,1,0,1])
    print(cos)
