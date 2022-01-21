from cmath import log
from html import entities
import pardata
import spacy
from spacy.tokens import DocBin
from collections import Counter
from tqdm import tqdm
import pickle

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

print('Loading dataset...')
wikitext103_data = pardata.load_dataset('wikitext103')
print('done')

MIN_TOKEN_FREQ = 10

def preprocess(dataset, doc_bin_filename):

    # Split raw text dataset into individual lines 
    lines = dataset.splitlines()

    # Lowercase
    lower_lines = [line.lower() for line in lines]

    # lower_lines = lower_lines[:1000]

    # Accumulate docs for pickling
    # We can only pickle (serialize) docs unfortunately
    doc_bin = DocBin(store_user_data=True)

    for doc in tqdm(nlp.pipe(lower_lines), total=len(lower_lines)):
        doc_bin.add(doc)

    # Save docs with tokens
    with open(f'{doc_bin_filename}_doc_bin.pickle', 'wb') as handle:
        pickle.dump(doc_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)

def tokens_from_serialized_docs(filename):
    # Read docs with tokens

    with open(f'{filename}_doc_bin.pickle', 'rb') as handle:
        print('Loading docs...')
        doc_bin = pickle.load(handle)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print('done')

        # Accumulate unigram POS tags
        doc_tokens = []
        text_tokens = []

        for doc in docs:
            for token in doc:
                # append single token with text, PoS tag and more to list of all tokens
                doc_tokens.append(token)

                # append just token text to a list to later count term frequencies
                text_tokens.append(token.text)
        
        # Remove infrequent terms
        # Gather token frequencies
        tokens_freqs = dict(Counter(text_tokens))

        # Filter out tokens that occur less than the minimum word frequency
        print('Filtering infrequent tokens...')
        frequent_tokens = list(filter(lambda token: tokens_freqs[token.text] >= 10, tqdm(doc_tokens)))
        print('done')

        return frequent_tokens

if __name__ == '__main__':
    # preprocess(wikitext103_data['train'], 'wikitext')
    tokens = tokens_from_serialized_docs('wikitext')
    print(f'Sample token: \nText: {tokens[40].text} \nPoS tag: {tokens[40].tag_}')
