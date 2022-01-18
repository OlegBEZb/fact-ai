from html import entities
import pardata
import spacy
from collections import Counter
from tqdm import tqdm

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

def preprocess(dataset):

    # Split raw text dataset into individual lines 
    lines = dataset.splitlines()

    # Lowercase
    lower_lines = [line.lower() for line in lines]

    # lower_lines = lower_lines[:1000]

    # Accumulate unigram POS tags
    doc_tokens = []
    text_tokens = []

    for doc in tqdm(nlp.pipe(lower_lines), total=len(lower_lines)):
        for token in doc:

            # append single token with text, PoS tag and more to list of all tokens
            doc_tokens.append(token)

            # append just token text to a list to later count term frequencies
            text_tokens.append(token.text)

    # Remove infrequent terms

    # Gather token frequencies
    tokens_freqs = dict(Counter(tqdm(text_tokens)))

    # Filter out tokens that occur less than the minimum word frequency
    frequent_tokens = list(filter(lambda token: tokens_freqs[token.text] >= 10, tqdm(doc_tokens)))

    return frequent_tokens

if __name__ == '__main__':
    tokens = preprocess(wikitext103_data['train'])
    print(f'Sample token: \nText: {tokens[40].text} \nPoS tag: {tokens[40].tag_}')

