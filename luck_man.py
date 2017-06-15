import pickle,json, random

def loader():
    with open('data/db/filterel4000.json') as f:
        lines = f.readlines()
        luck_number = random.randint(0, len(lines))
        luck_man = lines[luck_number]
        print(luck_man)
        # print(luck_number)
        # data_obj = []
        # for data in lines:
        #     json_data = json.loads(data)
        #     print(json_data)

if __name__ == '__main__':
    loader()