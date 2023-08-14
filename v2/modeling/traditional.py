from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys, os, pickle, collections, random, functools, time, typing, math
import imblearn
import numpy as np
import pandas as pd

DATA_DIR = '../data/cleaned'
OUTPUT_DIR = 'D:/data'
INPUT_COLUMN = 'text'
OUTPUT_COLUMN = 'classification'
K = 10 #k for k-fold xvalidation
METHODS = {'smote': imblearn.over_sampling.SMOTE, 
           'aug_fine': None,
           'none': None,
           'over': imblearn.over_sampling.RandomOverSampler, 
           'under': imblearn.under_sampling.RandomUnderSampler, 
           'aug': None,}
LIMIT = 10000000 #upper limit to number of samples in each dataset. Used for testing.
K_NEIGHBORS = 6 #number of neighbors for SMOTE
GENERATIVE_LLM_STRING = 'gpt2'
LLM_INITIALIZER = GPT2LMHeadModel.from_pretrained
GENERATIVE_LLM_MODEL = LLM_INITIALIZER('gpt2')
MIN_COUNT = 3
MODEL_DIR = f'D:/models/{GENERATIVE_LLM_STRING}'
GENERATIVE_LLM_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
models = ['traditional',]
fine_models = dict()

@functools.cache
def generate(text: str) -> str:
    '''Perform the generative LLM text generation.'''
    tokenized = GENERATIVE_LLM_TOKENIZER.encode(text, max_length=15, truncation=True, return_tensors='pt')
    generation = GENERATIVE_LLM_MODEL.generate(tokenized, max_length=25, do_sample=True, length_penalty=1, 
                                    temperature=random.random()*3, top_k=5, pad_token_id=GENERATIVE_LLM_TOKENIZER.eos_token_id)
    text = GENERATIVE_LLM_TOKENIZER.decode(generation[0], skip_special_tokens=True)
    text = text.replace('\n', ' ')
    return text

@functools.cache
def fine_generate(text: str, id: str) -> str:
    '''Perform the generative LLM text generation with a fine-tuned model.'''
    tokenized = GENERATIVE_LLM_TOKENIZER.encode(text, max_length=15, truncation=True, return_tensors='pt')
    generation = fine_models[id].generate(tokenized, max_length=25, do_sample=True, length_penalty=1, 
                                temperature=random.random()*3, top_k=5, pad_token_id=GENERATIVE_LLM_TOKENIZER.eos_token_id)
    text = GENERATIVE_LLM_TOKENIZER.decode(generation[0], skip_special_tokens=True)
    text = text.replace('\n', ' ')
    return text

def aug_resample(text: list[str], classifications: list[int]) -> tuple[list[str], list[int]]:
    '''Perform augmented LLM resampling given a list of inputs and outputs.'''
    output_text, output_classes = text.copy(), classifications.copy()
    counts = collections.Counter(classifications)
    maximum = max(counts.values())
    counts = {key: -(counts[key]-maximum) for key in counts}
    for key in counts:
        temp_text = [item for i, item in enumerate(text) if classifications[i] == key]
        output_text += [generate(temp_text[random.randint(0, len(temp_text)-1)]) for _ in range(counts[key])]
        output_classes += [key,] * counts[key]
    return output_text, output_classes

def fine_resample(text: list[str], classifications: list[int], id: str) -> tuple[list[str], list[int]]:
    '''Perform fine-tuned augmented LLM resampling given a list of inputs and outputs.'''
    output_text, output_classes = text.copy(), classifications.copy()
    counts = collections.Counter(classifications)
    maximum = max(counts.values())
    counts = {key: -(counts[key]-maximum) for key in counts}
    for key in counts:
        temp_text = [item for i, item in enumerate(text) if classifications[i] == key]
        output_text += [fine_generate(temp_text[random.randint(0, len(temp_text)-1)], id) for _ in range(counts[key])]
        output_classes += [key,] * counts[key]
    return output_text, output_classes

def smote_resample(inputs: np.ndarray, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Resample with SMOTE.'''
    resampler = METHODS['smote']()
    try:
        return resampler.fit_resample(inputs, outputs)
    except ValueError: #if the number in a particular class is not enough, random resample until it has the proper number
        value_counter = collections.Counter(outputs)
        needs_more = {key: value_counter[key] for key in value_counter if value_counter[key] < K_NEIGHBORS}
        for key in needs_more:
            temp_input = [row for i, row in enumerate(inputs) if outputs[i] == key]
            inputs = np.vstack((inputs, np.array([random.choice(temp_input) for _ in range(K_NEIGHBORS-needs_more[key])])))
            outputs = np.concatenate((outputs, np.array([key for _ in range(K_NEIGHBORS-needs_more[key])])))

    return resampler.fit_resample(inputs, outputs)

def just_vectorize(text: list[str], outputs: np.ndarray, method: typing.Optional[str], name: typing.Optional[str]) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    '''Only vectorize a list of strings with no resampling.'''
    split_text = ' '.join(text).split()
    wordlist = list(set(split_text))
    num_words = len(wordlist)
    all_counter = collections.Counter(split_text)
    wordlist = [word for word in wordlist if all_counter[word] >= MIN_COUNT]
    all_counter = {key: math.log(num_words/all_counter[key]) for key in all_counter}
    counters = [(collections.Counter(sentence.split()), len(sentence)) for sentence in text]
    sentences = np.array([[all_counter[word] * (0 if word not in counters[i][0] else counters[i][0][word]/counters[i][1]) for word in wordlist] for i, sentence in enumerate(text)])
    outputs = [outputs,]
    sentences = [(sentences, outputs),]
    return sentences, sentences.copy()

def aug_fine_resample_vectorize(df: pd.DataFrame, method: str, name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    '''Resample a text with a finetuned LLM.'''
    trains = [(list(df[INPUT_COLUMN]), list(df[OUTPUT_COLUMN])) for _ in models]
    tests = trains.copy()
    output_trains = []
    output_tests = []
    jump = len(trains[0][0]) // K
    for i, (inputs, outputs) in enumerate(trains):
        model = LLM_INITIALIZER(f'{MODEL_DIR}/{name}/{name}-{i}')
        fine_models[f'{name}-{i}'] = model
        spliced_inputs = [inputs[i*jump:(i+1)*jump] for i in range(K)]
        spliced_outputs = [outputs[i*jump:(i+1)*jump] for i in range(K)]
        tuple_list = [fine_resample(spliced_input, spliced_output, f'{name}-{i}') for spliced_input, spliced_output in zip(spliced_inputs, spliced_outputs)]
        all_inputs, all_outputs = [], []
        for temp_tuple_list in tuple_list:
            all_inputs += temp_tuple_list[0]
            all_outputs += temp_tuple_list[1]
        all_inputs = just_vectorize(all_inputs, all_inputs, None, None)[0][0]
        output_trains.append((all_inputs, np.array(all_outputs)))
        output_tests.append((just_vectorize(tests[i][0], tests[i][0], None, None)[0][0], tests[i][1]))
        del fine_models[f'{name}-{i}']
    return output_trains, output_tests

def aug_resample_vectorize(df: pd.DataFrame, method: str, name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    '''Perform an augmented resampling then vectorization on a `DataFrame`.'''
    trains = [(list(df[INPUT_COLUMN]), list(df[OUTPUT_COLUMN])) for _ in models]
    tests = trains.copy()
    output_trains = []
    output_tests = []
    jump = len(trains[0][0]) // K
    for i, (inputs, outputs) in enumerate(trains):
        spliced_inputs = [inputs[i*jump:(i+1)*jump] for i in range(K)]
        spliced_outputs = [outputs[i*jump:(i+1)*jump] for i in range(K)]
        tuple_list = [aug_resample(spliced_input, spliced_output) for spliced_input, spliced_output in zip(spliced_inputs, spliced_outputs)]
        all_inputs, all_outputs = [], []
        for temp_tuple_list in tuple_list:
            all_inputs += temp_tuple_list[0]
            all_outputs += temp_tuple_list[1]
        all_inputs = just_vectorize(all_inputs, all_inputs, None, None)[0][0]
        output_trains.append((all_inputs, np.array(all_outputs)))
        output_tests.append((just_vectorize(tests[i][0], tests[i][0], None, None)[0][0], tests[i][1]))
    return output_trains, output_tests

def vectorize_resample(df: pd.DataFrame, method: str, name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    '''Perform a vectorization then traditional resampling on a `DataFrame`.'''
    trains, tests = just_vectorize(list(df[INPUT_COLUMN]), df[OUTPUT_COLUMN].to_numpy(), method, name)
    output_trains = []
    jump = len(trains[0][0]) // K
    for inputs, outputs in trains:
        outputs = outputs[0]
        spliced_inputs = [inputs[i*jump:(i+1)*jump] for i in range(K)]
        spliced_outputs = [outputs[i*jump:(i+1)*jump] for i in range(K)]
        tuple_list = [smote_resample(spliced_input, spliced_output) if method == 'smote' else METHODS[method]().fit_resample(spliced_input, 
                                                                            spliced_output) for (spliced_input, 
                                                                            spliced_output) in zip(spliced_inputs, spliced_outputs)]
        all_inputs, all_outputs = tuple_list[0][0], tuple_list[0][1]
        for temp_tuple_list in tuple_list[1:]:
            all_inputs = np.vstack((all_inputs, temp_tuple_list[0]))
            all_outputs = np.concatenate((all_outputs, temp_tuple_list[1]))
        output_trains.append((all_inputs, all_outputs))
    return output_trains, tests

def resample_vectorize_file(filename: str, name: str) -> list[tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]]:
    '''Perform the entire resampling and vectorization process on a file with all resampling methods.'''
    functions = {'aug': aug_resample_vectorize, 'aug_fine': aug_fine_resample_vectorize, 'none': just_vectorize}
    df = pd.read_csv(filename)
    df[INPUT_COLUMN] = df[INPUT_COLUMN].astype(str)
    df = df.iloc[:LIMIT]
    outputs = []
    for resampling_method in METHODS:
        start = time.time()
        resampling_function = functions[resampling_method] if resampling_method in functions else vectorize_resample
        if resampling_method == 'none':
            arrays = resampling_function(list(df[INPUT_COLUMN]), df[OUTPUT_COLUMN].to_numpy(), resampling_method, name[:-4])
        else:
            arrays = resampling_function(df, resampling_method, name[:-4])
        outputs.append(arrays)
        end = time.time()
        print(f'Finished {resampling_method} method on {filename} dataset in {end-start:.2f} seconds.', end=' ')
    return outputs

def write_to_file(arrays: list[tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]], filename: str) -> None:
    '''
    Write a list of lists of vectorized and resampled data to files.
    The first dimension corresponds to a resampling method, the second to an LLM, and the third to a dataset.
    '''
    method_strings = list(METHODS.keys())
    for method_index, resampling_method_arrays in enumerate(arrays):
        trains, tests = resampling_method_arrays
        for model_index, (train_array, test_array) in enumerate(zip(trains, tests)):
            with open(f'{OUTPUT_DIR}/{method_strings[method_index]}/{models[model_index]}/train-input-{filename[:-4]}', 'wb') as f:
                pickle.dump(train_array[0], f)
            with open(f'{OUTPUT_DIR}/{method_strings[method_index]}/{models[model_index]}/train-output-{filename[:-4]}', 'wb') as f:
                pickle.dump(train_array[1], f)

            with open(f'{OUTPUT_DIR}/{method_strings[method_index]}/{models[model_index]}/test-input-{filename[:-4]}', 'wb') as f:
                pickle.dump(test_array[0], f)
            with open(f'{OUTPUT_DIR}/{method_strings[method_index]}/{models[model_index]}/test-output-{filename[:-4]}', 'wb') as f:
                pickle.dump(test_array[1], f)

def main():
    files = [file for file in os.listdir(DATA_DIR) if file.endswith('.csv')]
    for file in files:
        write_to_file(resample_vectorize_file(f'{DATA_DIR}/{file}', file), file)
        print('Written to file.\n')

if __name__ == '__main__':
    main()