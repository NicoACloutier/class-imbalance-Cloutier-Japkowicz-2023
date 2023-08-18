import os, string
import nltk
import pandas as pd

RAW_DATA_DIR = '../data/raw'
CLEAN_DATA_DIR = '../data/cleaned'
INPUT_COLUMN = 'text'
OUTPUT_COLUMN = 'classification'
MAX_LENGTH = 100000000
MAX_TEXT_LENGTH = 250

def delete_punctuation(text: str) -> str:
    '''Remove punctuation from a text.'''
    return ''.join(char for char in text if char not in string.punctuation)

def clean(text: str) -> str:
    '''Clean a text for processing.'''
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text.replace(f' {stopword} ', ' ')
    return text if len(text) <= MAX_TEXT_LENGTH else text[:MAX_TEXT_LENGTH]

def main():
    files = os.listdir(RAW_DATA_DIR)
    for file in files:
        df = pd.read_csv(f'{RAW_DATA_DIR}/{file}')
        df = df.dropna()
        df[INPUT_COLUMN] = df[INPUT_COLUMN].astype(str)
        df[INPUT_COLUMN] = df[INPUT_COLUMN].apply(clean)
        df = df.dropna()
        to_sample = min([len(df), MAX_LENGTH])
        df = df.sample(to_sample)
        df.to_csv(f'{CLEAN_DATA_DIR}/{file}', index=False)

if __name__ == '__main__':
    main()