from filter_compare import MainCompare
import pickle

if __name__ == '__main__':
    result_data = {}
    for i in [0.25, 0.50, 0.75]:
        main_obj = MainCompare()
        result_data[str(i)] = main_obj.main_process(i)

    pickle.dump(result_data, open('data/result/resize.data', 'wb'))
