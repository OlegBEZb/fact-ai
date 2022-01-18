from gensim.models import KeyedVectors

from word_embedding import download_word2vec, create_vocabulary
from seeds import get_seed

def WEAT(word_embedding, X, Y, A, B):
    """A bias measurement algorithm called Word Embedding Association Test
    (Caliskan et al., 2017). It defines a vector based on the difference between
    two target sets, and then measures the cosine similarity of a set of
    attribute words to that vector.
    The formula for calculating the WEAT score is as follows:

    s(X, Y, A, B) = sum_{x \in X} s(x, A, B) - sum_{y \in Y} s(y, A, B)
    where s(w, A, B) is the difference in average cosine similarities between
    query w and each term in A and w and each term in B.

    Input:
        X - first set of seeds
        Y - second set of seeds
        A - first attribute term
        B - second attribute term
    Output:
        s - float representing the WEAT score
    """

    score_x, score_y = 0., 0.

    def s(w, A, B):
        avg_a, avg_b = 0., 0.

        for i in range(len(A)):
            try:
                sim_a = word_embedding.similarity(w, A[i])
                sim_b = word_embedding.similarity(w, B[i])
            except KeyError:
                continue
            avg_a += sim_a
            avg_b += sim_b
        avg_a /= len(A)
        avg_b /= len(B)
        return avg_a - avg_b

    for x in X:
        score_x += s(x, A, B)
    for y in Y:
        score_y += s(y, A, B)

    s = score_x - score_y
    return s

if __name__ == '__main__':
    download_word2vec()
    word_embedding = create_vocabulary()
    """Start of replicating Figure 4
    """
    def print_WEAT(X, Y):
        WEAT_score = WEAT(word_embedding, X[0], Y[0], ['good'], ['bad'])
        print(f"WEAT score for '{X[1]}' and '{Y[1]}': {WEAT_score}")

    X = get_seed('family-Caliskan_et_al_2017')
    Y = get_seed('career-Caliskan_et_al_2017')
    print_WEAT(X, Y)

    X = get_seed('asian-Manzini_et_al_2019')
    Y = get_seed('caucasian-Manzini_et_al_2019')
    print_WEAT(X, Y)

    X = get_seed('female_2-Caliskan_et_al_2017')
    Y = get_seed('male_2-Caliskan_et_al_2017')
    print_WEAT(X, Y)

    X = get_seed('female_definition_words_1-Zhao_et_al_2018')
    Y = get_seed('male_definition_words_1-Zhao_et_al_2018')
    print_WEAT(X, Y)

    X = get_seed('names_asian-Garg_et_al_2018')
    Y = get_seed('names_chinese-Garg_et_al_2018')
    print_WEAT(X, Y)

    X = get_seed('names_white-Garg_et_al_2018')
    Y = get_seed('names_black-Garg_et_al_2018')
    print_WEAT(X, Y)
