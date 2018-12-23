import pickle
import numpy as np

with open('Annotations.pkl', 'rb') as f:
    annotations = pickle.load(f)


def generate(key):  # key in ['train', :'val', 'test']
    n = 0
    for i in range(len(annotations[key])):
        n += annotations[key][i]['people']['num']

    dic = {}
    i = j = k = 0
    for _ in range(n):
        if k == annotations[key][j]['people']['num']:
            j += 1
            k = 0
        if key == 'train' and np.isnan(annotations[key][j]['people'][k]['continuous'][0]):
            k += 1
            continue
        if key != 'train' and annotations[key][j]['people'][k]['categories']['num'] == 0:
            k += 1
            continue
        dic[i] = (j, k)
        k += 1
        i += 1
    return dic


mapping = {}
mapping['train'] = generate('train')
mapping['val'] = generate('val')
mapping['test'] = generate('test')


with open('Mapping.pkl', 'wb') as f:
    pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)
