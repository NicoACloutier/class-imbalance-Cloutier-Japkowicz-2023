import pandas as pd
import nltk
import math
import re
import collections
import string

WORD_COLUMN = 'Words'
PERCENT_COLUMN = 'Percent Appearance'
TFIDF = '.\\frequencies\\all.csv'

vector_distance = lambda v1, v2: math.sqrt(sum([(v1[i]-v2[i])**2 for (i, _) in enumerate(v1)])) #distance between two vectors equation
dot = lambda v1, v2: sum([v1[i]*v2[i] for (i, _) in enumerate(v1)])
mag = lambda v: math.sqrt(sum([item**2 for item in v]))
cosine_similarity = lambda v1, v2: dot(v1, v2)/mag(v1)*mag(v2)
delete_punctuation = lambda x: ''.join([char for char in x if char not in string.punctuation])

#put text counter into df
def to_df(text_counter, column1, column2):
    keys = [key for key in text_counter]
    values = [text_counter[key] for key in text_counter]
    output_df = pd.DataFrame([keys, values], [column1, column2]).transpose()
    return output_df
    
def get_text_counter(text):
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    
    #get a counter of the percentage of the text taken up by each word in the text
    split_text = text.split()
    text_counter = collections.Counter(split_text)
    length = len(split_text)
    for key in text_counter:
        text_counter[key] = text_counter[key] / length
    
    text_counter = collections.OrderedDict(text_counter.most_common())
    
    return text_counter

#get a counter of the percentage of a text taken up by the words in that text.
def get_counter(type, df):

    #put the text of the rows of a certain type in a string, and get rid of stopwords
    df = df[df['type'] == type]
    text = ' '.join(list(df['text']))
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    
    #get a counter of the percentage of the text taken up by each word in the text
    split_text = text.split()
    text_counter = collections.Counter(split_text)
    length = len(split_text)
    for key in text_counter:
        text_counter[key] = text_counter[key] / length
    
    text_counter = collections.OrderedDict(text_counter.most_common())
    
    return text_counter

#input frequency df for tweet, list of comparison dfs for dataset,
#and names of column1 (words) and column2 (frequencies),
#output the vector of the text and list of vectors for each comparison df
def to_vector(text_df, comparison_dict, column1, column2):
    comparison_vector_dict = dict()
    text_df = text_df.sort_values(column1) #sort alphabetically (so they're in the same order)
    text_vector = list(text_df[column2])
    words = list(text_df[column1]) #words in tweet

    for key in comparison_dict:
        comparison_df = comparison_dict[key]
        comparison_words = list(comparison_df[column1]) #words in dataset
        
        #add words that appear in tweet and not in dataset to dataset
        #(they all get frequency scores of 0)
        non_overlap_words = [word for word in words if word not in comparison_words] #words in tweet, not in dataset
        zeroes = [0 for word in non_overlap_words]
        non_overlap_df = pd.DataFrame([non_overlap_words, zeroes], [column1, column2]).transpose()
        comparison_df = pd.concat([comparison_df, non_overlap_df])
        
        #delete values in datasetthat aren't in text
        comparison_df = comparison_df.loc[comparison_df[column1].isin(words)]
        comparison_df = comparison_df.sort_values(column1) #sort alphabetically (so they're in the same order)
        
        comparison_vector = list(comparison_df[column2])
        comparison_vector_dict[key] = comparison_vector
    
    return text_vector, comparison_vector_dict
    
#do same as above, but with tfidf
def to_vector_tfidf(text_df, comparison_dict, tfidf_df, column1, column2):
    comparison_vector_dict = dict()
    text_df = text_df.sort_values(column1) #sort alphabetically (so they're in the same order)
    text_vector = list(text_df[column2])
    words = list(text_df[column1]) #words in tweet
    
    tfidf_words = list(tfidf[column1])
    non_overlap_words = [word for word in words if word not in tfidf_words] #words in tweet, not in dataset
    zeroes = [0 for word in non_overlap_words]
    non_overlap_df = pd.DataFrame([non_overlap_words, zeroes], [column1, column2]).transpose()
    comparison_df = pd.concat([comparison_df, non_overlap_df])
    
    #delete values in dataset that aren't in text
    comparison_df = comparison_df.loc[comparison_df[column1].isin(words)]
    comparison_df = comparison_df.sort_values(column1) #sort alphabetically (so they're in the same order)
    
    tfidf_vector = list(comparison_df[column2])

    for key in comparison_dict:
        comparison_df = comparison_dict[key]
        comparison_words = list(comparison_df[column1]) #words in dataset
        
        #add words that appear in tweet and not in dataset to dataset
        #(they all get frequency scores of 0)
        non_overlap_words = [word for word in words if word not in comparison_words] #words in tweet, not in dataset
        zeroes = [0 for word in non_overlap_words]
        non_overlap_df = pd.DataFrame([non_overlap_words, zeroes], [column1, column2]).transpose()
        comparison_df = pd.concat([comparison_df, non_overlap_df])
        
        #delete values in dataset that aren't in text
        comparison_df = comparison_df.loc[comparison_df[column1].isin(words)]
        comparison_df = comparison_df.sort_values(column1) #sort alphabetically (so they're in the same order)
        
        comparison_vector = list(comparison_df[column2])
        comparison_vector_dict[key] = comparison_vector
    
    text_vector = [item * tfidf[i] for (i, item) in text_vector]
    for key in comparison_vector_dict:
        comparison_vector_dict[key] = [item * tfidf[i] for (i, item) in comparison_vector_dict[key]]
    
    return text_vector, comparison_vector_dict

#compare the text vector with the database vectors
def compare_vectors(text_vector, comparison_vector_dict):
    distances = [] #list of tuples in format (distance, key)
    for key in comparison_vector_dict:
        compare_vector = comparison_vector_dict[key]
        distance = vector_distance(text_vector, compare_vector)
        distances.append((distance, key))
    closest = min(distances)[1]
    return closest

#compare vectors using cosine similarity
def cosine_compare_vectors(text_vector, comparison_vector_dict):
    similarities = [] #list of tuples in format (distance, key)
    for key in comparison_vector_dict:
        compare_vector = comparison_vector_dict[key]
        similarity = cosine_similarity(text_vector, compare_vector)
        similarities.append((similarity, key))
    closest = max(similarities)[1]
    return closest

#put it all together
def compare_text(text, keys, df_dict=None, column1=WORD_COLUMN, column2=PERCENT_COLUMN):
    text_counter = get_text_counter(text)
    text_df = to_df(text_counter, column1, column2)
    
    #make dictionary with given keys
    if not df_dict:
        df_dict = dict()
        for key in keys:
            df_dict[key] = pd.read_csv(f'frequencies\\{key}.csv')
    
    text_vector, comparison_vector_dict = to_vector(text_df, df_dict, column1, column2)
    closest = compare_vectors(text_vector, comparison_vector_dict)
    return closest
    
#compare text using cosine similarity
def cosine_compare_text(text, keys, df_dict=None, column1=WORD_COLUMN, column2=PERCENT_COLUMN):
    text_counter = get_text_counter(text)
    text_df = to_df(text_counter, column1, column2)
    
    #make dictionary with given keys
    if not df_dict:
        df_dict = dict()
        for key in keys:
            df_dict[key] = pd.read_csv(f'frequencies\\{key}.csv')
    
    text_vector, comparison_vector_dict = to_vector(text_df, df_dict, column1, column2)
    closest = compare_vectors(text_vector, comparison_vector_dict)
    
    return closest

#compare text with tfidf
def tfidf_compare_text(text, keys, tfidf_df, df_dict=None, column1=WORD_COLUMN, column2=PERCENT_COLUMN):
    text_counter = get_text_counter(text)
    text_df = to_df(text_counter, column1, column2)
    
    #make dictionary with given keys
    if not df_dict:
        df_dict = dict()
        for key in keys:
            df_dict[key] = pd.read_csv(f'frequencies\\{key}.csv')
    
    text_vector, comparison_vector_dict = to_vector(text_df, df_dict, column1, column2)
    closest = compare_vectors(text_vector, comparison_vector_dict)
    
    return closest

#returns True if classified as antisemitic, False otherwise
def is_antisemitic(text, df_dict=None):
    return 'antisemitic' == compare_text(text, ['baseline', 'antisemitic'], df_dict=df_dict)

#same as above, but with tfidf
def is_antisemitic_tfidf(text, tfidf_df=pd.DataFrame(), df_dict=None):
    if tfidf_df.empty: tfidf_df = pd.read_csv(TFIDF)
    return 'antisemitic' == tfidf_compare_text(text, ['baseline', 'antisemitic'], tfidf_df=tfidf_df, df_dict=df_dict)

#classifies text with cosine similarity
def cosine_is_antisemitic(text, df_dict=None):
    return 'antisemitic' == cosine_compare_text(text, ['baseline', 'antisemitic'], df_dict=df_dict)

#returns type of antisemitism predicted
def type(text, df_dict=None):
    types = {'1': 'political', '2': 'economic', '3': 'religious', '4': 'racial'}
    return types[compare_text(text, ['1', '2', '3', '4'], df_dict=df_dict)]

#classify type of antisemitism using tf-idf text representations
def type_tfidf(text, tfidf_df=pd.DataFrame(), df_dict=None):
    types = {'1': 'political', '2': 'economic', '3': 'religious', '4': 'racial'}
    if tfidf_df.empty: tfidf_df = pd.read_csv(TFIDF)
    return types[tfidf_compare_text(text, ['1', '2', '3', '4'], tfidf_df=tfidf_df, df_dict=df_dict)]

#returns type of antisemitism predicted with cosine similarity
def cosine_type(text, df_dict=None):
    types = {'1': 'political', '2': 'economic', '3': 'religious', '4': 'racial'}
    return types[cosine_compare_text(text, ['1', '2', '3', '4'], df_dict=df_dict)]