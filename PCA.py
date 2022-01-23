from gensim.models import KeyedVectors

from sklearn.decomposition import PCA
from scipy import spatial
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

from word_embedding import download_word2vec, create_vocabulary
from seeds import get_seed

def create_bias_subspace(word_embedding, X, Y, shuffle=False, n_components=None):
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
        n_components - how many components used to calculate PCA, use None to calculate
                       explainability per component, and use 1 to calculate coherence
    Returns:
        pca - PCA object representing the bias subspace
    """
    if not len(X) == len(Y):
        print(f'Error: expected paired seed words, but received sets of different sizes ({len(X)} and {len(Y)})')
        return None

    if shuffle:
        #When shuffle is set to True, consider all possible pairings between the seeds
        pairs = list(itertools.product(X, Y))
    else:
        pairs = list(zip(X, Y))

    halves = []
    for x, y in pairs:
        #Only use pairs when both are present in the word embeddings
        try:
            embed_x = word_embedding[x]
            embed_y = word_embedding[y]
        except KeyError:
            continue

        mean = embed_x - embed_y

        half1, half2 = mean - embed_x, mean - embed_y
        halves.append(half1)
        halves.append(half2)

    halves = np.stack(halves)
    pca = PCA(n_components=n_components)
    pca.fit(halves)

    return pca

def calculate_coherence(word_embedding, X, Y):
    """
    Function to calculate coherence between two seed pair sets.
    It first creates a subspace, using the first pca component of half vectors,
    created using the create_bias_subspace function.
    Then it will rank all words in seed sets X and Y.
    Then it will find coherence(X,Y) = |mean_rank(X) - mean_rank(Y)|,
    where the mean_rank is normalized to [0,1]

    Parameters:
        word_embedding - dictionary containing the word embeddings
        X - Tuple of set of paired seed words and category
        Y - Tuple of set of paired seed words and category
    Output:
        coherence - value between 0 and 1, indicating coherence between input seed sets
    """
    pca = create_bias_subspace(word_embedding, X[0], Y[0], shuffle=False, n_components=1)

    #Create a ranking dictionary ranking the similarity with the first pca component
    all_sims = word_embedding.similar_by_vector(pca.components_[0], topn=10000000, restrict_vocab=None)
    word_to_sims_index = {}
    for i, sim_tuple in enumerate(all_sims):
        word_to_sims_index[sim_tuple[0]] = i

    #get all rankings of the words in X and Y
    rank_x, rank_y = [], []
    for i in range(len(X[0])):
        try:
            word_x_rank = word_to_sims_index[X[0][i]]
            word_y_rank = word_to_sims_index[Y[0][i]]
        except KeyError:
            continue

        rank_x.append(word_x_rank)
        rank_y.append(word_y_rank)

    #normalize the mean rank to [0,1]
    avg_x_rank = np.mean(rank_x)
    avg_y_rank = np.mean(rank_y)

    min_value = min(avg_x_rank, avg_y_rank)
    max_value = max(avg_x_rank, avg_y_rank)
    avg_x = avg_x_rank / (avg_x_rank + avg_y_rank)
    avg_y = avg_y_rank / (avg_x_rank + avg_y_rank)

    #calculate coherence
    coherence = abs(avg_x - avg_y)

    print(f"coherence between '{X[1]}' and '{Y[1]}' is {coherence}")
    return coherence

def plot(word_embedding):
    """Plot explained variance of each component of the PCA subset of various
    shuffled and unshuffled seed pairs, as described in figure 3 of the bad seeds paper
    """
    # Seeds from Bolukbasi et al. (2016)
    pca = create_bias_subspace(word_embedding, get_seed('definitional_female-Bolukbasi_et_al_2016')[0],
                               get_seed('definitional_male-Bolukbasi_et_al_2016')[0])
    EV_gender_order = pca.explained_variance_ratio_

    pca = create_bias_subspace(word_embedding, get_seed('definitional_female-Bolukbasi_et_al_2016')[0],
                               get_seed('definitional_male-Bolukbasi_et_al_2016')[0], shuffle=True)
    EV_gender_random = pca.explained_variance_ratio_

    # Seeds from Kozlowski et al. (2019)
    pca = create_bias_subspace(word_embedding, get_seed('upperclass-Kozlowski_et_al_2019')[0],
                               get_seed('lowerclass-Kozlowski_et_al_2019')[0])
    EV_social_class_order = pca.explained_variance_ratio_

    pca = create_bias_subspace(word_embedding, get_seed('upperclass-Kozlowski_et_al_2019')[0],
                               get_seed('lowerclass-Kozlowski_et_al_2019')[0], shuffle=True)
    EV_social_class_random = pca.explained_variance_ratio_

    # Seeds from Garg et al. (2018)
    pca = create_bias_subspace(word_embedding, get_seed('names_chinese-Garg_et_al_2018')[0],
                               get_seed('names_hispanic-Garg_et_al_2018')[0])
    EV_names_order = pca.explained_variance_ratio_

    pca = create_bias_subspace(word_embedding, get_seed('names_chinese-Garg_et_al_2018')[0],
                               get_seed('names_hispanic-Garg_et_al_2018')[0], shuffle=True)
    EV_names_random = pca.explained_variance_ratio_

    #Plot PCA components
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,3))

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

    X = np.arange(10)
    names_data = [EV_names_order[0:10], EV_names_random[0:10]]

    ax3.bar(X + 0.0, names_data[0], color = 'b', width = 0.3, label='original order')
    ax3.bar(X + 0.3, names_data[1], color = 'g', width = 0.3, label='shuffled')
    ax3.legend()
    ax3.set_title('Chinese-Hispanic Name Pairs')
    plt.show()

def create_coherence_table(word_embedding):
    """Replicating Figure 4
    """
    calculate_coherence(word_embedding, get_seed('career-Caliskan_et_al_2017'), get_seed('family-Caliskan_et_al_2017'))
    calculate_coherence(word_embedding, get_seed('asian-Manzini_et_al_2019'), get_seed('caucasian-Manzini_et_al_2019'))
    calculate_coherence(word_embedding, get_seed('female_2-Caliskan_et_al_2017'), get_seed('male_2-Caliskan_et_al_2017'))
    calculate_coherence(word_embedding, get_seed('female_definition_words_1-Zhao_et_al_2018'), get_seed('male_definition_words_1-Zhao_et_al_2018'))
    calculate_coherence(word_embedding, get_seed('names_asian-Garg_et_al_2018'), get_seed('names_chinese-Garg_et_al_2018'))
    calculate_coherence(word_embedding, get_seed('names_black-Garg_et_al_2018'), get_seed('names_white-Garg_et_al_2018'))

if __name__ == '__main__':
    download_word2vec()
    word_embedding = create_vocabulary()
    create_coherence_table(word_embedding)
    plot(word_embedding)
