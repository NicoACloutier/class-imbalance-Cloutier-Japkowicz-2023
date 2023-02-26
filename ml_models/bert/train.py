import pandas as pd
from transformers import AutoTokenizer, AutoModelForClassification

MODEL_NAME = 'bert-base-uncased'
DATA_FILE = '..\\..\\data\\train.csv'
NUM_SPLITS = 10
TASKS = {'classification': 'binary', 'type_of_antisemitism': 'type'}

tokenize = lambda sentence, tokenizer: tokenizer(sentence, max_length=20, padding=True, truncation=True, return_vectors='pt')['input_ids']

#fine-tune a model using an input and output column
def train(df, tokenizer, input_column, output_column):
    inputs = [tokenize(item, tokenizer) for item in list(df[input_column])]
    outputs = [tokenize(item, tokenizer) for item in list(df[output_column])]
    data = [{'input_ids': item[0], 'labels': item[1]} for item in zip(inputs, outputs)]

    training_args = TrainingArguments(
        output_dir='test_trainer',
        evaluation_strategy='epoch',
        num_train_epochs=3,
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
    )

    trainer.train()

    return model

#get the predictions of a model on an input column
def test(df, model, tokenizer, input_column):
    inputs = [tokenize(item, tokenizer) for item in list(df[input_column])]
    answers = model(inputs)
    return answers

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_FILE)
    jump = len(df) // NUM_SPLITS
    all_answers = []

    for task in ['classification', 'type_of_antisemitism']:
        for split in range(NUM_SPLITS):
            #prepare data for training and testing
            test_df = df.iloc[split*jump:(split+1)*jump]
            train_df = df.drop(test_df.index)
            train_df = train_df.reset_index()
            test_df = test_df.reset_index()

            #train and get answers for evaluation
            model = train(train_df, tokenizer, 'text', task)
            answers = test(test_df, tokenizer, 'text', task)
            all_answers += answers

        
        #write answers to file for analysis
        output_df = pd.read_csv(f'..\\{TASKS[task]}_predictions.csv')
        output_df['BERT'] = all_answers
        output_df.to_csv(f'..\\{TASKS[task]_predictions.csv}')

if __name__ == '__main__':
    main()
