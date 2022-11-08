import pandas as pd
import nltk
import collections
import string
import re
import math

to_zero = lambda x: 0 if math.isnan(x) else x
delete_punctuation = lambda x: ''.join([char for char in x if char not in string.punctuation])

def get_counter(type, df, baseline=None):
    df = df[df['type'] == type]
    text = ' '.join(list(df['text']))
    text = text.lower()
    for stopword in nltk.corpus.stopwords.words('english'):
        text = re.sub(f' {stopword} ', ' ', text)
    
    split_text = text.split()
    text_counter = collections.Counter(split_text)
    length = len(split_text)
    for key in text_counter:
        text_counter[key] = text_counter[key] * 100 / length
    
    if baseline:
        for key in text_counter:
            value = 0 if key not in baseline else baseline[key]
            text_counter[key] = text_counter[key] - value
    
    text_counter = collections.OrderedDict(text_counter.most_common())
    
    return text_counter

def main():

    df = pd.read_csv('antisemitism_dataset.csv')
    
    df['type_of_antisemitism'] = df['type_of_antisemitism'].apply(to_zero)
    df['type_of_antisemitism'] = df['type_of_antisemitism'].astype(int)
    df['classification'] = df['classification'].astype(int)
    df['type'] = df['classification'] + df['type_of_antisemitism']
    df['text'] = df['text'].astype(str) #convert column to strings
    df['text'] = df['text'].apply(delete_punctuation)
    
    baseline = get_counter(0, df)
    types = range(1, 5)
    for type in types:
        type_counter = get_counter(type, df, baseline)
        formatted_string = '\n'.join([f'{key}: {type_counter[key]:.3f}' for key in type_counter])
        with open(f'counters\\{type}.txt', 'w', encoding='utf8') as f:
            f.write(formatted_string)

if __name__ == '__main__':
    main()