import pandas as pd
from sklearn import tree, svm, naive_bayes
import string, collections, math
import nltk
import time
import numpy as np
import xgboost
import imblearn

BASIC = '..\\data'
MODEL_DIR = '.\\models'
NUMBER_PARTITIONS = 10
COLUMNS = ['Algorithm', 'Representation', 'Task', 'Score', 'Partition']
BERT_NAME = 'bert-base-uncased'
OUTPUT_DIR = '.\\predictions'
THRESHOLD_PCT = 0.5 #percent of dataset word must take up to be considered for dataset

methods = {'RandomUnder': imblearn.under_sampling.RandomUnderSampler,}

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
            dict1[key] = list(dict1[key]) + list(dict2[key])
    return dict1

#remove stopwords from text, put in lowercase, and delete punctuation
def clean(text):
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    return text
    
#fit a model and test
def fit_test(model, model_name, train_input, train_output, test_input):
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)
    return predictions
    
#add the classification and types columns to each other in the place of the type column, replace NAs with 0
def combine_answers(df):
    df = df.fillna(0)
    columns = list(df.columns.values)
    type, classification = columns.index('type'), columns.index('classification')
    df['type'] = df.apply(lambda x: x[type] + x[classification], axis=1)
    return df
    
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
    
#/WORD REPRESENTATIONS
    
#train all algorithms on a single representation method and single partition,
#update report df
def train_representations(train_input, train_output, test_input, test_output, 
                          rep_method, algorithms, partition, task, resampling_method, stratified_column):
    
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
    
    #take care of column used for resampling
    stratified_column = temp_train_output if stratified_column.empty else stratified_column.values.tolist()
    
    if resampling_method == 'ADASYN':
        temp_train_input, temp_train_output = methods[resampling_method](sampling_strategy='minority').fit_resample(temp_train_input, stratified_column)
    elif resampling_method != 'none':
        temp_train_input, temp_train_output = methods[resampling_method]().fit_resample(temp_train_input, stratified_column)

    if 'binary' in task:
        temp_train_output = [int(x != 0) for x in temp_train_output] #turn multi-class classifications into binary if binary task

    finish = time.time()
    period = finish - begin
    print(f'{rep_name} representations finished in {period:.2f} seconds.')
    
    temp_predictions['Actual'] = temp_test_output
    
    for algorithm in algorithms:
        start = time.time()
        
        algo_name = algorithm['name']
        algo_model = algorithm['model']
    
        predictions = list(fit_test(algo_model, f'{algo_name}-{rep_name}-{task}', 
                                    temp_train_input, temp_train_output,
                                    temp_test_input))
        
        temp_predictions[f'{algo_name}-{rep_name}'] = predictions
        
        end = time.time()
        elapsed = end - start
        print(f'Finished training {algo_name} model with {rep_name} representation, predicting {task} in {elapsed:.2f} seconds.')
    
    return temp_predictions
    
#given a single partition, train and test each model with each rep method
def train_and_test(train_input, train_output, test_input, 
                   test_output, partition, task, resampling_method, stratified_column):
    temp_predictions = dict()
        
    #get wordlist
    all_text = ' '.join(list(train_input) + list(test_input))
    all_text_split = all_text.split()
    all_text_counter = collections.Counter(all_text_split)
    wordlist = list(set(all_text_split))
    num_words = len(wordlist)
    
    words_to_delete = []
    for word in wordlist:
        if all_text_counter[word]*100/num_words < THRESHOLD_PCT:
            words_to_delete.append(word)
    
    for word in words_to_delete:
        wordlist.remove(word)
    num_words = len(wordlist)
    
    word_dict = dict()
    for word in wordlist:
        value = math.log(num_words/all_text_counter[word])
        word_dict[word] = value
    
    rep_methods = [{'name': 'bow', 'function': bow, 'wordlist': wordlist}, #methods for representing text
                   {'name': 'freq', 'function': freq, 'wordlist': wordlist},
                   {'name': 'tfidf', 'function': tfidf, 'wordlist': word_dict},]
    
    algorithms = [{'name': 'decision-tree', 'model': tree.DecisionTreeClassifier()}, #algorithms for classification
                  {'name': 'svm', 'model': svm.SVC()},
                  {'name': 'naive-bayes', 'model': naive_bayes.GaussianNB()},
                  {'name': 'xgboost', 'model': xgboost.XGBClassifier()}]
    
    for rep_method in rep_methods:
        temp_predictions = dict_concat(train_representations(train_input, train_output, test_input, test_output, 
                                       rep_method, algorithms, partition, task, resampling_method, stratified_column), 
                                       temp_predictions)
    
    return temp_predictions

#perform the entire process of training all models with all representations across all partitions,
#only for one task
def process_fit_test(df, test_df, output, resampling_method, stratified_column):
    prediction_dict = dict()

    length = len(df)
    jump = length // NUMBER_PARTITIONS #value increased each partition
    test_jump = len(test_df) // NUMBER_PARTITIONS
    
    if stratified_column:
        stratified_column = df[stratified_column]
    else:
        stratified_column = pd.Series(dtype='float64')
    
    for partition in range(NUMBER_PARTITIONS):
        print(f'\nPARTITION NUMBER {partition+1}.\n')
    
        begin = partition * jump
        end = (partition+1) * jump
        
        test_begin = partition * test_jump
        test_end = (partition+1) * test_jump
        
        to_drop = df.iloc[begin:end]
        train_df = df.drop(to_drop.index)
        temp_test_df = test_df.iloc[test_begin:test_end]
        if stratified_column.empty:
            temp_stratified_column = stratified_column.copy()
        else:
            temp_stratified_column = stratified_column.drop(to_drop.index)
        
        temp_test_df = temp_test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        train_input = train_df['text']
        train_output = train_df[output]
        
        test_input = temp_test_df['text']
        test_output = temp_test_df[output]
        
        prediction_dict = append_dict_values(train_and_test(train_input, train_output, test_input, test_output, partition, 
                                                            output, resampling_method, temp_stratified_column), prediction_dict)
    
    predictions_df = pd.DataFrame(prediction_dict)
    
    return predictions_df

def main():

    df = pd.read_csv(f'{BASIC}\\augmented_dataset.csv')
    df['text'] = df['text'].apply(clean).astype(str)
    df = df[df['text'].apply(lambda x: len(x.split()) >= 1)]
    df = df.fillna(0)
    
    test_df = pd.read_csv(f'{BASIC}\\antisemitism_dataset.csv')
    test_df['text'] = test_df['text'].apply(clean).astype(str)
    test_df = test_df[test_df['text'].apply(lambda x: len(x.split()) >= 1)]
    test_df['type'] = test_df['type_of_antisemitism']
    test_df = test_df.fillna(0)
    
    #binary task
    binary_predictions_df = process_fit_test(df, test_df, 'classification', 'RandomUnder', None)
    binary_predictions_df.to_csv(f'{OUTPUT_DIR}\\binary_predictions-aug.csv', index=False)
    
    #stratified binary task
    binary_predictions_df = process_fit_test(df, test_df, 'classification', 'RandomUnder', 'type')
    binary_predictions_df.to_csv(f'{OUTPUT_DIR}\\strat-binary_predictions-aug.csv', index=False)
    
    #4 type classification task
    temp_df = df[df['classification'] == 1]
    temp_test_df = test_df[test_df['classification'] == 1]
    type4_predictions_df = process_fit_test(temp_df, temp_test_df, 'type', 'RandomUnder', None)
    type4_predictions_df.to_csv(f'{OUTPUT_DIR}\\4type_predictions-aug.csv', index=False)
    
    #5 type classification task
    df = combine_answers(df)
    test_df = combine_answers(test_df)
    type5_predictions_df = process_fit_test(df, test_df, 'type', 'RandomUnder', None)
    type5_predictions_df.to_csv(f'{OUTPUT_DIR}\\5type_predictions-aug.csv', index=False)
    
if __name__ == '__main__':
    main()

