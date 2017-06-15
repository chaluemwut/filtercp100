from result_printing import print_all_result
import pickle
import matplotlib.pyplot as plt
import numpy as np

# def perf_boxplot():
#     all_result = pickle.load(open('data/result/result400_v2.data', 'rb'))
#     plt.boxplot([all_result['perf_text'], all_result['perf_ml'], all_result['perf_ml_word']])
#     fig = plt.figure(1, figsize=(9, 6))
#     ax = fig.add_subplot(111)
#     ax.set_xticklabels(['Topic Feature', 'Text Features', 'Social Media Features'])
#     plt.show()

all_result = pickle.load(open('data/newresult/result/cos_result.obj', 'rb'))

def new_main_plot1():
    plt.boxplot([all_result['f1_topic_lst'], all_result['f1_text_lst'], all_result['f1_social_lst']])
    fig = plt.figure(1, figsize=(2, 2))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['Topic Features', 'Text Features', 'Social Features'])
    plt.ylabel('F1-Score value')
    plt.show()

def new_main_plot2():
    plt.boxplot([all_result['f1_social_lst'], all_result['f1_social_and_text_lst']])
    fig = plt.figure(1, figsize=(1, 1))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['Social Features', 'Social and Text Features'])
    plt.ylabel('F1-Score value')
    plt.show()

def all_result():
    cos_result = pickle.load(open('data/newresult/result/cos_result.obj', 'rb'))
    ml_result = pickle.load(open('data/newresult/10/all_result.obj', 'rb'))
    plt.boxplot([
                 cos_result['f1_topic_lst'], \
                 cos_result['f1_text_lst'], \
                 cos_result['f1_social_lst'], \
                 cos_result['f1_social_and_text_lst'], \
                 ml_result['f1_topic_lst'], \
                 ml_result['f1_text_lst'], \
                 ml_result['f1_social_lst'], \
                 ml_result['f1_social_and_text_lst']
                    ])
    plt.ylabel('F1-Score value')
    plt.show()

def all_result_new():
    # cos_result = pickle.load(open('data/all_result/all_result_cos.obj', 'rb'))
    ml_result = pickle.load(open('data/all_result/all_result_ml.obj', 'rb'))
    plt.boxplot([
                # cos_result['f1_topic_lst'], \
                # cos_result['f1_text_lst'], \
                #
                # cos_result['f1_topic_and_text_lst'], \
                # cos_result['f1_social_lst'], \
                #
                # cos_result['f1_social_and_text_lst'], \
                # cos_result['f1_topic_and_social_lst'], \
                # cos_result['f1_topic_text_social_lst'],\

                ml_result['f1_topic_lst'], \
                ml_result['f1_text_lst'], \

                ml_result['f1_topic_and_text_lst'], \
                ml_result['f1_social_lst'],\

                ml_result['f1_topic_and_social_lst'],\
                ml_result['f1_social_and_text_lst'],\
                ml_result['f1_topic_text_social_lst']
                ])
    plt.show()

if __name__ == '__main__':
    all_result_new()