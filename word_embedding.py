import urllib.request
import os
import numpy as np

def download_word2vec(dir='data', file_name='word_embedding.txt'):
    """Download the 300D truncated Google News word2vec vectors
    You can find the original file here: https://code.google.com/archive/p/word2vec/

    Input:
        dir - string of the name of the directory to save the embeddings to
        file_name - string of the name of the file to save the embeddings to
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

    output_file = os.path.join(dir, file_name)
    if not os.path.isfile(output_file):
        print('Downloading... ', end='', flush=True)
        urllib.request.urlretrieve('https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt', "data/word_embedding.txt")
        print('done')

def create_vocabulary(file='data/word_embedding.txt'):
    """
    Input:
        file - name of file with embeddings
    Output:
        embedding - dictionary with words as keys and 300D word embedding vectors as value
    """
    word2vec_data = []
    with open(file) as f:
        for line in f:
            word2vec_data.append(line)

    embedding = {}

    for data_set in word2vec_data:
        word_and_vec = data_set.split(' ')

        #Seperate the word and the vector of each line
        word = word_and_vec[0]
        vec = [float(num) for num in word_and_vec[1:]]
        embedding[word] = np.array(vec)
    return embedding
