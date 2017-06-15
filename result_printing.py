def print_all_result(all_result):
    print('********** performance result')
    for ml, ml_word, topic, perf_text in zip(all_result['perf_ml'], all_result['perf_ml_word'], all_result['perf_topic'], all_result['perf_text']):
        print('{},{},{},{}'.format(ml, ml_word, topic, perf_text))

    print('********** training time')
    for t_ml, t_ml_word, t_topic, t_text in zip(all_result['time_train_ml'], all_result['time_train_ml_word'], all_result['time_train_topic'], all_result['time_train_text']):
        print('{},{},{},{}'.format(t_ml, t_ml_word, t_topic, t_text))

    print('********** testing time')
    for p_ml, p_ml_word, p_topic, p_text in zip(all_result['time_predict_ml'], all_result['time_predict_ml_word'], all_result['time_predict_topic'], all_result['time_predict_text']):
        print('{},{},{},{}'.format(p_ml, p_ml_word, p_topic, p_text))