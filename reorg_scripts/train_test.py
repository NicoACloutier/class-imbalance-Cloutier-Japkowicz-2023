import pandas as pd

BASIC = '..\\data'

def main():
    df = pd.read_csv(f'{BASIC}\\antisemitism_dataset.csv')
    df = df.dropna(subset=['text'])
    train_df = df.sample(frac=0.9)
    test_df = df.drop(index=train_df.index)
    
    train_df.to_csv(f'{BASIC}\\train.csv', index=False)
    test_df.to_csv(f'{BASIC}\\test.csv', index=False)

if __name__ == '__main__':
    main()
