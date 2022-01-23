from cmath import log
from html import entities
import pardata
import spacy
from spacy.tokens import DocBin
from collections import Counter
from tqdm import tqdm
import pickle
# import datetime
# from pynytimes import NYTAPI
import os

nyt = NYTAPI("xdAb2YGzt44pk62Pug6W0AOrKgEidCga", parse_dates=True)

# Only use the PoS tagger, or processing will take very long
nlp = spacy.load('en_core_web_sm', disable=[
    'parser',
    'entity',
    'ner',
    'entity_linker',
    'entity_ruler',
    'textcat',
    'textcat_multilabel',
    'morphologizer',
    'senter',
    'sentencizer',
    'tok2vec',
    'transformers'
])


MIN_TOKEN_FREQ = 10

def preprocess(dataset, doc_bin_filename, doc_bin_size=50000):

    # Split raw text dataset into individual lines 
    lines = dataset.splitlines()

    # Lowercase
    lower_lines = [line.lower() for line in lines]

    # lower_lines = lower_lines[]

    # Accumulate docs for pickling
    # We can only pickle (serialize) docs unfortunately
    doc_bin = DocBin(store_user_data=True)
    doc_bin_num = 0
    iter = 0

    for doc in tqdm(nlp.pipe(lower_lines), total=len(lower_lines)):
        iter += 1
        doc_bin.add(doc)
        if iter >= doc_bin_size:
            # Save docs with tokens
            with open(f'./preprocessed/{doc_bin_filename}_doc_bin_{doc_bin_num}.pickle', 'wb') as handle:
                pickle.dump(doc_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
            doc_bin_num += 1
            iter = 0
            doc_bin = DocBin(store_user_data=True)

 
def tokens_from_serialized_docs(dataset_name):

    # Accumulate unigram POS tags
    doc_tokens = []
    text_tokens = []

    # Read all parts of the docs
    filenames = os.listdir('./preprocessed')
    filenames.sort()
    print(filenames)
    print('Processing files...')
    for filename in tqdm(filenames):

        # Read docs with tokens
        with open(f'./preprocessed/{filename}', 'rb') as handle:
            # print('Loading docs...')
            doc_bin = pickle.load(handle)
            docs = list(doc_bin.get_docs(nlp.vocab))

            for doc in docs:
                for token in doc:
                    # append single token with text, PoS tag and more to list of all tokens
                    doc_tokens.append(token)

                    # append just token text to a list to later count term frequencies
                    text_tokens.append(token.text)
            # print('done')

        handle.close()

    # Remove infrequent terms
    # Gather token frequencies
    tokens_freqs = dict(Counter(text_tokens))

    # Filter out tokens that occur less than the minimum word frequency
    print('Filtering infrequent tokens...')
    frequent_tokens = list(filter(lambda token: tokens_freqs[token.text] >= MIN_TOKEN_FREQ, tqdm(doc_tokens)))
    print('done')

    return frequent_tokens

def find_nyt_articles(date_start, date_end):
    articles = nyt.article_search(
        dates = {
            "begin": date_start,
            "end": date_end
        },
        options = {
            "results": 100,
            "sort": "oldest",
            "sources": [
                "New York Times",
            ]
        }
    )

    print(len(articles))
    return articles

if __name__ == '__main__':
    # print('Loading dataset...')
    # wikitext103_data = pardata.load_dataset('wikitext103')
    # print('done')
    # preprocess(wikitext103_data['train'], 'wikitext')
    
    tokens = tokens_from_serialized_docs('wikitext')
    print(f'Sample token: \nText: {tokens[40].text} \nPoS tag: {tokens[40].tag_}')

    # load NYT data using NYT Developers api
    # articles = find_nyt_articles(datetime.datetime(2016, 4, 15), datetime.datetime(2016, 4, 15))
    # print(articles)
    # print(len(articles))
