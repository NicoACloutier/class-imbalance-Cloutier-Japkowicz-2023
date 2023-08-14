import os, pickle, typing, collections
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import datasets
import pandas as pd

DATA_DIR = '../data/cleaned'
LLM = 'gpt2'
OUTPUT_DIR = f'D:/models/{LLM}'
INPUT_COLUMN = 'text'
OUTPUT_COLUMN = 'classification'
K = 10
LR = 1.372e-4
MODEL_INITIALIZER = AutoModelForCausalLM.from_pretrained
GENERATIVE_LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM)
GENERATIVE_LLM_TOKENIZER.pad_token = GENERATIVE_LLM_TOKENIZER.eos_token

def finetune(df: pd.DataFrame) -> dict[int, transformers.Trainer]:
    '''Finetune a single split of a df'''
    output_dict = dict()
    num_epochs = 3 if len(df) < 2000 else 1
    counter = collections.Counter(df[OUTPUT_COLUMN])
    for key in [key for key in counter if counter[key] != max(counter.values())]:
        temp_df = df[df[OUTPUT_COLUMN] == key]
        model = MODEL_INITIALIZER(LLM)
        collator = DataCollatorForLanguageModeling(tokenizer=GENERATIVE_LLM_TOKENIZER, mlm=False)
        dataset = [GENERATIVE_LLM_TOKENIZER.encode(sample, max_length=25, truncation=True, padding='max_length', return_tensors='pt') for sample in list(temp_df[INPUT_COLUMN])]
        
        train_args = TrainingArguments(
            f'{OUTPUT_DIR}/saves/', overwrite_output_dir=True,
            learning_rate=LR, weight_decay=0.01, save_strategy='epoch', save_steps=200, 
            report_to=None, logging_steps=50, remove_unused_columns=False, num_train_epochs=num_epochs)
        trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator,)
        trainer.train()
        output_dict[key] = trainer
    
    return output_dict

def write_to_file(trainers: dict[int, transformers.Trainer], dataset: str, split: int) -> None:
    '''Write a finetuned model to file'''
    for trainer in trainers:
        trainers[trainer].save_model(f'{OUTPUT_DIR}/{dataset}/{dataset}-{split}-{trainer}')

def finetune_models(dataset: str) -> None:
    '''Finetune models on all splits and write to file'''
    df = pd.read_csv(f'{DATA_DIR}/{dataset}')
    df[INPUT_COLUMN] = df[INPUT_COLUMN].astype(str)
    jump = len(df) // K
    for split in range(K):
        temp_df = df.drop(range(split*jump, (split+1)*jump))
        write_to_file(finetune(temp_df), dataset[:-4], split)

def main():
    datasets = ['antisemitism_two.csv', 'antisemitism_four.csv', 'antisemitism_five.csv']
    for dataset in datasets:
        finetune_models(dataset)

if __name__ == '__main__':
    main()