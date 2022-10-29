import re
import pandas as pd

select = lambda value_list, selection_list, minimum_value: [value for (i, value) in enumerate(value_list) if selection_list[i] > minimum_value]

def main():
    with open('raw.txt', 'r', encoding='utf8') as f:
        text = f.read()
        text = re.sub('\n[A-Z]\n', '', text) #delete lines with only single uppercase letter (these lines denote the beginning of a new letter, not useful)
        words = re.findall('(?<=\n).+?(?= \– )', text) #get words, in format (\nWORD - ).
        definitions = re.findall('(?<= \– ).+?(?=\n)', text) #get definitions, in format ( - DEFINITION\n).
    
    with open('4chanwords.txt', 'r', encoding='utf8') as f:
        text = f.read()
        text = re.sub('\n.(?=\n)', '', text) #delete single-character lines
        temp_words = re.findall('(?<=\n).+?(?=\n)', text) #get words
        words += temp_words #add to total words
        definitions += ['None'] * len(temp_words) #add 'None' definitions to words
    
    df = pd.read_csv('cleaned_data.csv') #read data
    
    appearances_list = [] #list of appearances of each word
    averaged_values = {'hateful': [], 'offensive': [],
                       'sentiment': [], 'violent': []} #values to be averaged for each word and recorded in df
    for i, word in enumerate(words):
        line_df = df[df['text'].str.contains(f'[,\.\s\?]{word}s?[,\.\s\?]', case=False)] #make df of lines with word appearing in it
        appearances = len(line_df) #get number of appearances for word
        appearances_list.append(appearances)
        if appearances > 0:
            for key in averaged_values:
                avg = sum(line_df[key]) / appearances
                averaged_values[key].append(avg)
    
    #get rid of words/definitions that have appearance values of 0
    words = select(words, appearances_list, 0)
    definitions = select(definitions, appearances_list, 0)
    appearances_list = select(appearances_list, appearances_list, 0)
    
    #write data to df
    word_df = pd.DataFrame([words, definitions, appearances_list], ["word", "definition", "appearances"]).transpose()
    for key in averaged_values:
        word_df[key] = averaged_values[key]
    
    #sort descending by # of appearances
    word_df = word_df.sort_values('appearances', ascending=False)
    
    word_df.to_csv('words.csv', index=False)

if __name__ == '__main__':
    main()