import re
import pandas as pd

def main():
    with open('raw.txt', 'r', encoding='utf8') as f:
        text = f.read()
        text = re.sub('\n[A-Z]\n', '', text) #delete lines with only single uppercase letter (these lines denote the beginning of a new letter, not useful)
        words = re.findall('(?<=\n).+?(?= \– )', text) #get all of the words in the text, in the format (\nWORD - ).
        definitions = re.findall('(?<= \– ).+?(?=\n)', text) #get all of the definitions in the text, in the format ( - DEFINITION\n).
    
    word_series = pd.DataFrame([words, definitions], ["Words", "Definitions"]).transpose()
    word_series.to_csv('words.csv', index=False)

if __name__ == '__main__':
    main()