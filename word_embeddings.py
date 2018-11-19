import bcolz, pickle
import numpy as np

glove_path = "word_embeddings"
"""
modified from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

input should just be the filename with no extensions or directory
dumps some compressed files into the disk for easier loading/lookup of word embeddings later

usage example: preprocessing.extract_word_embeddings("glove.6B.50d")
"""
def extract_word_embeddings(filename):
    words = []
    idx = 0
    word2idx = {}
    dims = None
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/{filename}.dat', mode='w')

    with open(f'{glove_path}/{filename}.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            if idx == 1:
                dims = len(vect)
            vectors.append(vect)
    vectors = bcolz.carray(vectors[1:].reshape((idx, dims)), rootdir=f'{glove_path}/{filename}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/{filename}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/{filename}_idx.pkl', 'wb'))

"""
taken from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

in: filename of preprocessed glove embeddings (ex: glove.6B.50d), must run extract_word_embeddings first
out: dict that u can use to look up vectors for words

loading example:
glove = preprocessing.get_glove_dict("glove.6B.50d")

lookup example:
glove["the"]
"""
def get_glove_dict(name):
    vectors = bcolz.open(f'{glove_path}/{name}.dat')[:]
    words = pickle.load(open(f'{glove_path}/{name}_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/{name}_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove