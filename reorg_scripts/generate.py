from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers
import pandas as pd
import string

INPUT_FILE = '..\\data\\antisemitism_dataset.csv'
OUTPUT_FILE = '..\\data\\augmented_dataset.csv'
NUM_TEMPS = 20

tokenize = lambda text, tokenizer: tokenizer.encode(text, max_length=15, truncation=True, return_tensors='pt')
delete_punctuation = lambda x: ''.join([char for char in x if char not in [string.punctuation]+['â€”&gt;']])

#multiply each item of an array a certain number of times
def multiply(list, times_over):
    final_list = []
    for item in list_array:
      final_list += [item for _ in range(times_over)]
    return final_list

def main():
    transformers.utils.logging.set_verbosity_error()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    df = pd.read_csv(INPUT_FILE)
    original_texts = list(df['text'].astype(str).apply(delete_punctuation))
    final_texts = []
    for text in original_texts:
        length = len(text)
        encoding = tokenize(text, tokenizer)
        for temp in range(NUM_TEMPS):
            temp /= (NUM_TEMPS/2)
            temp += 1
            generation = model.generate(encoding, max_length=25, do_sample=True, min_new_tokens=5, length_penalty=1, temperature=temp, top_k=10)
            text = tokenizer.decode(generation[0], skip_special_tokens=True)
            text = text.replace('\n', ' ')
            final_texts.append(text)
    
    classifications = multiply(list(df['classification'].astype(int)), NUM_TEMPS)
    types = multiply(list(df['type_of_antisemitism'].astype(int)), NUM_TEMPS)
    
    output_df = pd.DataFrame(data=[final_texts, classifications, types], columns=['text', 'classification', 'type_of_antisemitism'])
    output_df.to_csv(OUTPUT_FILE)

if __name__ == '__main__':
    main()