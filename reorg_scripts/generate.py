from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd

INPUT_FILE = '..\\data\\antisemitism_dataset.csv'
OUTPUT_FILE = '..\\data\\augmented_dataset.csv'
NUM_TEMPS = 20

tokenize = lambda text, tokenizer: tokenizer.encode(text, return_tensors='pt')

#multiply each item of an array a certain number of times
def multiply(list, times_over):
    array = np.vstack(np.array(list) * times_over)
    array = array.T
    array = np.concat(array)
    return array

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    df = pd.read_csv(INPUT_FILE)
    original_texts = list(df['text'].astype(str))
    final_texts = []
    for text in original_texts:
        encoding = tokenize(text, tokenizer)
        for temp in range(NUM_TEMPS):
            temp += (1/(NUM_TEMPS/2))
            temp /= (NUM_TEMPS/2)
            generation = model.generate(encoding, max_length=200, do_sample=True, temperature=temp)
            text = tokenizer.decode(generation[0], skip_special_tokens=True)
            final_texts += text
    
    classifications = multiply(list(df['classification'].astype(int)), NUM_TEMPS)
    types = multiply(list(df['type_of_antisemitism'].astype(int)), NUM_TEMPS)
    
    output_df = pd.DataFrame(data=[final_texts, classifications, types], columns=['text', 'classification', 'type_of_antisemitism'])
    output_df.to_csv(OUTPUT_FILE)

if __name__ == '__main__':
    main()