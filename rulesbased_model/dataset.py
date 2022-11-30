import pandas as pd
import rulesbased as rb
import time

#This script tests the rules-based models on the testing data.

DATA_DIR = '..\\data'
FREQ_DIR = 'frequencies'

def main():
    start = time.time()
    types = {'political': 0,
             'economic': 1,
             'religious': 2,
             'racial': 3}

    #get dataframes for bool is_antisemitic
    is_antisemitic_dict = {'baseline': pd.read_csv(f'{FREQ_DIR}\\baseline.csv'),
                           'antisemitic': pd.read_csv(f'{FREQ_DIR}\\antisemitic.csv')}
    
    tfidf_df = pd.read_csv('.\\frequencies\\all.csv')
    
    #get dataframes for types
    type_dict = dict()
    for x in range(1, 5):
        type_dict[str(x)] = pd.read_csv(f'{FREQ_DIR}\\{x}.csv')
    
    end = time.time()
    print(f'Setup completed in {end - start:.3f} seconds.')
    
    #apply to files
    files = ['test']
    for filename in files:
        file_start = time.time()
        df = pd.read_csv(f'{DATA_DIR}\\{filename}.csv')
        
        start = time.time()
        df['is_antisemitic'] = df['text'].apply(lambda x: int(rb.is_antisemitic(x, df_dict=is_antisemitic_dict)))
        end = time.time()
        print(f'Binary freq completed in {int((end - start)//60)} minutes and {(end-start)-(end-start)//60*60:.2f} seconds.')
        start = time.time()
        df['predicted_type'] = df['text'].apply(lambda x: types[rb.type(x, df_dict=type_dict)])
        end = time.time()
        print(f'Type freq completed in {int((end - start)//60)} minutes and {(end-start)-(end-start)//60*60:.2f} seconds.')
        
        start = time.time()
        df['tfidf-is_antisemitic'] = df['text'].apply(lambda x: int(rb.is_antisemitic_tfidf(x, tfidf_df=tfidf_df, df_dict=is_antisemitic_dict)))
        end = time.time()
        print(f'Binary tfidf completed in {int((end - start)//60)} minutes and {(end-start)-(end-start)//60*60:.2f} seconds.')
        start = time.time()
        df['tfidf-type'] = df['text'].apply(lambda x: types[rb.type_tfidf(x, tfidf_df=tfidf_df, df_dict=type_dict)])
        end = time.time()
        print(f'Type tfidf completed in {int((end - start)//60)} minutes and {(end-start)-(end-start)//60*60:.2f} seconds.')
        
        df.to_csv(f'{DATA_DIR}\\predicted\\{filename}-predicted.csv', index=False)
        file_end = time.time()
        print(f'File {filename} completed in {int((file_end - file_start)//60)} minutes and {(file_end-file_start)-(file_end-file_start)//60*60:.2f} seconds.')

if __name__ == '__main__':
    main()
