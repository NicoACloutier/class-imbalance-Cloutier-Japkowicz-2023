import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = 'bert-base-uncased'
DATA_FILE = '..\\..\\data\\train.csv'
NUM_SPLITS = 10
TASKS = {'classification': 'binary', 'type_of_antisemitism': 'type'}

tokenize = lambda sentence, tokenizer: tokenizer(sentence, max_length=20, padding='max_length', truncation=True, return_tensors='pt')['input_ids']

#fine-tune a model using an input and output column
def train(df, tokenizer, input_column, output_column):
    inputs = [tokenize(item, tokenizer) for item in list(df[input_column])]
    inputs = torch.cat(inputs)

    data = [{'input_ids': inputs[i], 'labels': df[output_column].iloc[i]} for i in range(inputs.shape[0])]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir='test_trainer',
        num_train_epochs=5,
        save_steps=10000,
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
    df[input_column] = df[input_column].astype(str)
    inputs = torch.cat([tokenize(item, tokenizer) for item in list(df[input_column])])
    answers = model(inputs)
    return answers.logits

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_FILE)
    df['text'] = df['text'].astype(str)
    jump = len(df) // NUM_SPLITS
    all_answers = torch.Tensor()

    for task in ['classification', 'type_of_antisemitism']:
    
        if task == 'type_of_antisemitism':
            df = df[df['classification'] == 1]

        for split in range(NUM_SPLITS):
            #prepare data for training and testing
            test_df = df.iloc[split*jump:(split+1)*jump]
            train_df = df.drop(test_df.index)
            train_df = train_df.reset_index()
            test_df = test_df.reset_index()

            #train and get answers for evaluation
            model = train(train_df, tokenizer, 'text', task)
            answers = test(test_df, model, tokenizer, task)
            all_answers = torch.cat([all_answers, answers])
        
        all_answers = np.argmax(all_answers, axis=1)
        
        #write answers to file for analysis
        output_df = pd.read_csv(f'..\\{TASKS[task]}_predictions.csv')
        output_df['BERT'] = all_answers
        output_df.to_csv(f'..\\{TASKS[task]}_predictions.csv')

if __name__ == '__main__':
    main()