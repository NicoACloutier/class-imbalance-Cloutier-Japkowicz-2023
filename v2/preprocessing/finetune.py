import os, pickle, typing
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

def finetune(df: pd.DataFrame) -> transformers.Trainer:
    '''Finetune a single split of a df'''
    model = MODEL_INITIALIZER(LLM)
    collator = DataCollatorForLanguageModeling(tokenizer=GENERATIVE_LLM_TOKENIZER, mlm=False)
    dataset = [GENERATIVE_LLM_TOKENIZER.encode(sample, max_length=25, truncation=True, padding='max_length', return_tensors='pt') for sample in list(df[INPUT_COLUMN])]
    
    train_args = TrainingArguments(
        f'{OUTPUT_DIR}/saves/', overwrite_output_dir=True,
        learning_rate=LR, weight_decay=0.01, save_strategy='epoch', save_steps=1, 
        report_to=None, logging_steps=5, remove_unused_columns=False)
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator,)
    trainer.train()
    
    return trainer

def write_to_file(trainer: transformers.Trainer, dataset: str, split: int) -> None:
    '''Write a finetuned model to file'''
    trainer.save_model(f'{OUTPUT_DIR}/{dataset}-{split}')

def finetune_models(dataset: str) -> None:
    '''Finetune models on all splits and write to file'''
    df = pd.read_csv(f'{DATA_DIR}/{dataset}')
    del df[OUTPUT_COLUMN]
    df = df.astype(str)
    jump = len(df) // K
    for split in range(K):
        temp_df = df.drop(range(split*jump, (split+1)*jump))
        write_to_file(finetune(temp_df), dataset[:-4], split)

def main():
    datasets = [file for file in os.listdir(DATA_DIR) if file.endswith('.csv')]
    for dataset in datasets:
        finetune_models(dataset)

if __name__ == '__main__':
    main()