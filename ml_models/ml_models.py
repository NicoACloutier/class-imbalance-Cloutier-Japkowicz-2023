import pandas as pd
from sklearn import tree, svm, naive_bayes
import string, collections, math
import nltk
import pickle
from transformers import AutoTokenizer, AutoModel
import time
import numpy as np

BASIC = '..\\data'
MODEL_DIR = '.\\models'
NUMBER_PARTITIONS = 10
COLUMNS = ['Algorithm', 'Representation', 'Task', 'Score', 'Partition']
BERT_NAME = 'bert-base-uncased'

value_or_zero = lambda x, dict: 0 if not x in dict else dict[x] #return dictionary value if key in dictionary, otherwise return 0
delete_punctuation = lambda x: ''.join([char for char in x if char not in string.punctuation]) #delete punctuation from string
sigmoid = lambda x: 1 / (1 + np.exp(-x))

#concatenate two dictionaries
def dict_concat(dict1, dict2):
    for key in dict2:
        dict1[key] = dict2[key]
    return dict1

#given two dictionaries with identical keys and list values,
#concatenate the corresponding values lists from the second to
#the first where the keys are the same
def append_dict_values(dict1, dict2):
    for key in dict1:
        if key in dict2:
            dict1[key] = dict1[key] + dict2[key]
    return dict1

#remove stopwords from text, put in lowercase, and delete punctuation
def clean(text):
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    return text
    
#fit a model, save to file, and test
def fit_save_test(model, model_name, train_input, train_output, test_input):
    model.fit(train_input, train_output)
    filename = f'{MODEL_DIR}\\{model_name}.sav'
    pickle.dump(model, open(filename, 'wb'))
    predictions = model.predict(test_input)
    return predictions
    
#WORD REPRESENTATIONS

#represent text with bag of words
def bow(text, wordlist):
    return [int(word in text) for word in wordlist]

#represent text with frequencies
def freq(text, wordlist):
    text = text.split()
    counter = collections.Counter(text)
    length = len(text)
    return [value_or_zero(word, counter)/length for word in wordlist]

#represent text with tf-idf
def tfidf(text, word_dict):
    text = text.split()
    counter = collections.Counter(text)
    length = len(text)
    frequencies = [value_or_zero(word, counter)/length for word in word_dict]
    return [word_dict[word] * frequencies[i] for (i, word) in enumerate(word_dict)]

#represent text with bert output, model_tokenizer is tuple with model and tokenizer
def bert(text, model_tokenizer):
    model, tokenizer = model_tokenizer
    text = tokenizer(text, max_length=20, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
    output = np.squeeze(model(text).pooler_output.detach().numpy())
    return sigmoid(output)
    
#/WORD REPRESENTATIONS
    
#train all algorithms on a single representation method and single partition,
#update report df
def train_representations(train_input, train_output, test_input, 
                          test_output, rep_method, algorithms, partition, task):
    
    temp_predictions = dict()
    
    begin = time.time()
    
    rep_name = rep_method['name']
    rep_func = rep_method['function']
    rep_list = rep_method['wordlist']
    
    #represent text with that method
    temp_train_input = train_input.apply(lambda x: rep_func(x, rep_list)).values.tolist()
    temp_train_output = train_output.values.tolist()
    temp_test_input = test_input.apply(lambda x: rep_func(x, rep_list)).values.tolist()
    temp_test_output = test_output.values.tolist()
    
    finish = time.time()
    period = finish - begin
    print(f'{rep_name} representations finished in {period:.2f} seconds.')
    
    temp_predictions['Actual'] = temp_test_output
    
    for algorithm in algorithms:
        start = time.time()
        
        algo_name = algorithm['name']
        algo_model = algorithm['model']
    
        predictions = list(fit_save_test(algo_model, f'{algo_name}-{rep_name}-{task}', 
                                         temp_train_input, temp_train_output,
                                         temp_test_input))
        
        temp_predictions[f'{algo_name}-{rep_name}-{task}'] = predictions
        
        end = time.time()
        elapsed = end - start
        print(f'Finished training {algo_name} model with {rep_name} representation, predicting {task} in {elapsed:.2f} seconds.')
    
    return temp_predictions
    
#given a single partition, train and test each model with each rep method
def train_and_test(train_input, train_output, test_input, 
                   test_output, partition, task):
    temp_predictions = dict()
    
    #get wordlist
    all_text = ' '.join(list(train_input) + list(test_input))
    all_text_split = all_text.split()
    all_text_counter = collections.Counter(all_text_split)
    wordlist = list(set(all_text_split))
    num_words = len(wordlist)
    word_dict = dict()
    for word in wordlist:
        value = math.log(num_words/all_text_counter[word])
        word_dict[word] = value
    
    model = AutoModel.from_pretrained(BERT_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    
    rep_methods = [{'name': 'bert', 'function': bert, 'wordlist': (model, tokenizer)},
                   {'name': 'bow', 'function': bow, 'wordlist': wordlist}, #methods for representing text
                   {'name': 'freq', 'function': freq, 'wordlist': wordlist},
                   {'name': 'tfidf', 'function': tfidf, 'wordlist': word_dict},]
    
    algorithms = [{'name': 'decision-tree', 'model': tree.DecisionTreeClassifier()}, #algorithms for classification
                  {'name': 'svm', 'model': svm.SVC()},
                  {'name': 'naive-bayes', 'model': naive_bayes.GaussianNB()},
                  {'name': 'naive-bayes-multinomial', 'model': naive_bayes.MultinomialNB()}]
    
    for rep_method in rep_methods:
        temp_predictions = dict_concat(train_representations(train_input, train_output, test_input, test_output, 
                                       rep_method, algorithms, partition, task), temp_predictions)
    
    return temp_predictions

#perform the entire process of training all models with all representations across all partitions,
#only for one task
def process_fit_test(df, output):
    prediction_dict = dict()

    length = len(df)
    jump = length // NUMBER_PARTITIONS #value increased each partition
    for partition in range(NUMBER_PARTITIONS):
        print(f'\nPARTITION NUMBER {partition+1}.\n')
    
        begin = partition * jump
        end = (partition+1) * jump
        
        test_df = df.iloc[begin:end]
        train_df = df.drop(test_df.index)
        test_df = test_df.reset_index()
        train_df = train_df.reset_index()

        train_input = train_df['text']
        train_output = train_df[output]
        test_input = test_df['text']
        test_output = test_df[output]
    
        prediction_dict = append_dict_values(train_and_test(train_input, train_output, test_input, 
                                                            test_output, partition, output), prediction_dict)
    
    predictions_df = pd.DataFrame(prediction_dict)
    
    return predictions_df

def main():

    #Data preprocessing
    df = pd.read_csv(f'{BASIC}\\antisemitism_dataset.csv')
    df['text'] = df['text'].apply(clean).astype(str)
    df = df[df['text'].apply(lambda x: len(x.split()) >= 1)]
    
    binary_predictions_df = process_fit_test(df, 'classification')
    binary_predictions_df.to_csv('binary_predictions.csv', index=False)
    
    df = df[df['classification'] == 1]
    type_predictions_df = process_fit_test(df, 'type_of_antisemitism')
    type_predictions_df.to_csv('type_predictions.csv', index=False)

if __name__ == '__main__':
    main()
