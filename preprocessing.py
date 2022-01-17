import pardata
import spacy
from collections import Counter

sp = spacy.load('en_core_web_sm')

wikitext103_data = pardata.load_dataset('wikitext103')

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
    for line in lower_lines:
        pos_tagged = sp(line)
        for token in pos_tagged:

            # append single token with text, PoS tag and more to list of all tokens
            doc_tokens.append(token)
            text_tokens.append(token.text)

    # Remove infrequent terms

    # Gather token frequencies
    tokens_freqs = dict(Counter(text_tokens))

    # Filter out tokens that occur less than the minimum word frequency
    frequent_tokens = list(filter(lambda token: tokens_freqs[token.text] >= 10, doc_tokens))

    return frequent_tokens

if __name__ == '__main__':
    tokens = preprocess(wikitext103_data['train'])
    print(f'Sample token: \nText: {tokens[40].text} \nPoS tag: {tokens[40].tag_}')
    
