import os
import numpy as np
import requests
from gensim.models import KeyedVectors

def download_file_from_google_drive(id, destination):
    """Download a file from google drive.

    Input:
        id - the id of the google drive file
        destination - string of the name of the file to save the downloaded file to
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_word2vec(dir='data', file_name='word_embedding.gz'):
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
        download_file_from_google_drive('0B7XkCwpI5KDYNlNUTTlSS21pQmM', output_file)
        print('done')

def create_vocabulary(dir='data', file_name='word_embedding.gz'):
    """
    Input:
        dir - string of the name of the directory to save the embeddings to
        file_name - string of the name of the file to save the embeddings to
    Output:
        embedding - KeyedVector with words as keys and 300D word embedding vectors as value
    """
    print('Creating embedding... ', end='', flush=True)
    file = os.path.join(dir, file_name)
    embedding = KeyedVectors.load_word2vec_format(file, binary=True)
    print('done... ')
    return embedding
