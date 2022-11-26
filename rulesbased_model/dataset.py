import pandas as pd
import rulesbased as rb

DATA_DIR = '..\\data'
FREQ_DIR = 'frequencies'

def main():
    types = {'political': 0,
             'economic': 1,
             'religious': 2,
             'racial': 3}

    #get dataframes for bool is_antisemitic
    is_antisemitic_dict = {'baseline': pd.read_csv(f'{FREQ_DIR}\\baseline.csv'),
                           'antisemitic': pd.read_csv(f'{FREQ_DIR}\\antisemitic.csv')}
    
    #get dataframes for types
    type_dict = dict()
    for x in range(1, 5):
        type_dict[str(x)] = pd.read_csv(f'{FREQ_DIR}\\{x}.csv')
    
    #apply to files
    files = ['normal_tweets', 'cleaned_data', 'test']
    for filename in files:
        df = pd.read_csv(f'{DATA_DIR}\\{filename}.csv')
        df['is_antisemitic'] = df['text'].apply(lambda x: int(rb.is_antisemitic(x, df_dict=is_antisemitic_dict)))
        df['predicted_type'] = df['text'].apply(lambda x: types[rb.type(x, df_dict=type_dict)])
        df['cos_is_antisemitic'] = df['text'].apply(lambda x: int(rb.cosine_is_antisemitic(x, df_dict=is_antisemitic_dict)))
        df['cos_predicted_type'] = df['text'].apply(lambda x: types[rb.cosine_type(x, df_dict=type_dict)])
        df.to_csv(f'{DATA_DIR}\\predicted\\{filename}-predicted.csv', index=False)

if __name__ == '__main__':
    main()
