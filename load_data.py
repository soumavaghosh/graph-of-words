from collections import Counter
from random import sample

def read_data(dataset):
    with open(f'{dataset}-train-all-terms.txt', 'r') as f:
        data_train = f.readlines()

    with open(f'{dataset}-test-all-terms.txt', 'r') as f:
        data_test = f.readlines()

    data_train = [x.replace('\n', '').split('\t') for x in data_train]
    data_test = [x.replace('\n', '').split('\t') for x in data_test]

    words = []
    classes = set()
    for i, j in data_train:
        words.extend(j.split(' '))
        classes.add(i)
    for i, j in data_test:
        words.extend(j.split(' '))
        classes.add(i)

    classes = list(classes)
    words_ctr = Counter(words)
    words = [i for i,j in words_ctr.items() if j>2]
    word2id = {j:i+1 for i,j in enumerate(words)}
    word2id[0] = None
    class2id = {j:i for i, j in enumerate(classes)}

    data_train = [(class2id[i], [word2id.get(x, len(word2id)) for x in j.split(' ')]) for i,j in data_train]
    data_test = [(class2id[i], [word2id.get(x, len(word2id)) for x in j.split(' ')]) for i, j in data_test]

    return data_train, data_test, len(word2id)+1, len(class2id)
