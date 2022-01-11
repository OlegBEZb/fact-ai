from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt

from word_embedding import download_word2vec, create_vocabulary

def create_bias_subspace(word_embedding, X, Y, shuffle=False):
    """
    Function to create a bias subspace.
    First the function will calculate for each pair of embeddings words
    the mean embedding vector, and then will calculate the two resulting half
    vectors from that mean to the two seed embedding vectors.
    Then it will calculate the bias subpace using PCA using those half vectors.

    Parameters:
        word_embedding - dictionary containing the word embeddings
        X - first set of paired seed words
        Y - second set of paired seed words
        Shuffle - when set to true, shuffle the pairs
    Returns:
        pca - PCA object representing the bias subspace
    """

    if not len(X) == len(Y):
        print('Error: expected paired seed words, but received sets of different sizes')

    if shuffle:
        random.shuffle(Y)

    halves = []
    for i in range(len(X)):
        x, y = X[i], Y[i]

        #Only use pairs when both are present in the word embeddings
        try:
            embed_x = word_embedding[x]
        except KeyError:
            continue

        try:
            embed_y = word_embedding[y]
        except KeyError:
            continue

        mean = embed_x - embed_y

        half1, half2 = mean - embed_x, mean - embed_y
        halves.append(half1)
        halves.append(half2)

    halves = np.stack(halves)
    pca = PCA()
    pca.fit(halves.T)

    return pca

if __name__ == '__main__':
    download_word2vec()
    word_embedding = create_vocabulary()

    #Run two PCA
    # Seeds from Bolukbasi et al. (2016)
    X = ['woman', 'girl', 'she', 'gal', 'mother', 'daughter', 'female', 'her', 'herself', 'Mary']
    Y = ['man', 'boy', 'he', 'guy', 'father', 'son', 'male', 'his', 'himself', 'John']
    pca = create_bias_subspace(word_embedding, X, Y)
    EV_gender_order = pca.explained_variance_ratio_

    pca = create_bias_subspace(word_embedding, X, Y, shuffle=True)
    EV_gender_random = pca.explained_variance_ratio_

    # Seeds from Kozlowski et al. (2019)
    X = ['rich', 'richer', 'richest', 'affluence', 'affluent', 'expensive', 'luxury', 'opulent']
    Y = ['poor', 'poorer', 'poorest', 'poverty', 'impoverished', 'inexpensive', 'cheap', 'needy']
    pca = create_bias_subspace(word_embedding, X, Y)
    EV_social_class_order = pca.explained_variance_ratio_

    pca = create_bias_subspace(word_embedding, X, Y, shuffle=True)
    EV_social_class_random = pca.explained_variance_ratio_

    #Plot PCA components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

    gender_data = [EV_gender_order[0:10], EV_gender_random[0:10]]
    X = np.arange(10)
    ax1.bar(X + 0.0, gender_data[0], color = 'b', width = 0.3, label='original order')
    ax1.bar(X + 0.3, gender_data[1], color = 'g', width = 0.3, label='shuffled')
    ax1.legend()
    ax1.set_title('Gender pairs')

    X = np.arange(8)
    social_data = [EV_social_class_order[0:8], EV_social_class_random[0:8]]

    ax2.bar(X + 0.0, social_data[0], color = 'b', width = 0.3, label='original order')
    ax2.bar(X + 0.3, social_data[1], color = 'g', width = 0.3, label='shuffled')
    ax2.legend()
    ax2.set_title('Social Class Pairs')
    plt.show()
