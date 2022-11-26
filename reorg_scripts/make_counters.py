import pandas as pd
import nltk
import collections
import string
import re
import math

COLUMN1 = 'Words'
COLUMN2 = 'Percent Appearance'

to_zero = lambda x: 0 if math.isnan(x) else x #set missing value to 0
delete_punctuation = lambda x: ''.join([char for char in x if char not in string.punctuation])

#put text counter into df
def to_df(text_counter, column1, column2):
    keys = [key for key in text_counter]
    values = [text_counter[key] for key in text_counter]
    output_df = pd.DataFrame([keys, values], [column1, column2]).transpose()
    output_df = output_df.sort_values(column2, ascending=False) #sort
    return output_df

#get a counter of the percentage of a text taken up by the words in that text.
#if a baseline counter is provided to compare, subtract it.
def get_counter(type, df, typename='type', baseline=pd.DataFrame()):

    #put the text of the rows of a certain type in a string, and get rid of stopwords
    df = df[df[typename] == type]
    text = ' '.join(list(df['text'].astype(str)))
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    
    #get a counter of the percentage of the text taken up by each word in the text
    split_text = text.split()
    text_counter = collections.Counter(split_text)
    length = len(split_text)
    for key in text_counter:
        text_counter[key] = text_counter[key] * 100 / length
    
    #if a baseline is provided, subtract the value from each word in the baseline
    #from the same word in the current text counter
    if not baseline.empty:
        for key in text_counter:
            value = 0 if key not in baseline else baseline[key]
            text_counter[key] = text_counter[key] - value
    
    text_counter = collections.OrderedDict(text_counter.most_common())
    
    return text_counter

def main():

    df = pd.read_csv('..\\data\\train.csv')
    
    #reconfigure/reorganize dataframe
    df['classification'] = df['classification'].astype(int)
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(delete_punctuation) #delete punctuation from the text
    
    #get counters and write to file for antisemitic text
    all_antisemitic = get_counter(1, df, typename='classification') #get a baseline percentage counter (this is the counter for the non-antisemitic text)
    all_df = to_df(all_antisemitic, COLUMN1, COLUMN2)
    all_df.to_csv('..\\rulesbased_model\\frequencies\\antisemitic.csv', index=False)
    
    #get counters and write to file for non-antisemitic text
    non_antisemitic_counter = get_counter(0, df, typename='classification')
    base_df = to_df(non_antisemitic_counter, COLUMN1, COLUMN2)
    base_df.to_csv('..\\rulesbased_model\\frequencies\\baseline.csv', index=False)
    
    #combine type of antisemitism and classification columns
    df['type_of_antisemitism'] = df['type_of_antisemitism'].apply(to_zero) #replace missing values with zero
    df['type_of_antisemitism'] = df['type_of_antisemitism'].astype(int)
    df['type'] = df['classification'] + df['type_of_antisemitism'] #type and classification are seperate columns, combine them

    types = range(1, 5) #remaining types of antisemitism
    for type in types:
        #make comparison counters between different types of antisemitism and
        #non-antisemitic text
        text_counter = get_counter(type, df, base_df) #make compared counter
        temp_df = to_df(text_counter, COLUMN1, 'Percent Increase')
        temp_df.to_csv(f'..\\data\\counters\\{type}.csv', index=False)
        
        #make raw counters to be used for the rules-based model
        frequency_counter = get_counter(type, df) #make frequency counter
        temp_df = to_df(frequency_counter, COLUMN1, COLUMN2)
        temp_df.to_csv(f'..\\rulesbased_model\\frequencies\\{type}.csv', index=False)

if __name__ == '__main__':
    main()