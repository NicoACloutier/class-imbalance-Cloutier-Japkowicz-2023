import re
import pandas as pd

def main():
    with open('raw.txt', 'r', encoding='utf8') as f:
        text = f.read()
        text = re.sub('\n[A-Z]\n', '', text)
        words = re.findall('(?<=\n).+?(?= \– )', text)
        definitions = re.findall('(?<= \– ).+?(?=\n)', text)
    
    word_series = pd.DataFrame([words, definitions], ["Words", "Definitions"]).transpose()
    word_series.to_csv('words.csv', index=False)

if __name__ == '__main__':
    main()