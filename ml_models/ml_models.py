import pandas as pd
from sklearn import tree, svm, naive_bayes
import string
import nltk
import collections
import math
import pickle

BASIC = '..\\data'
MODEL_DIR = '.\\models'
NUMBER_PARTITIONS = 10
COLUMNS = ['Algorithm', 'Representation', 'Task', 'Score', 'Partition']
report_df = pd.DataFrame(columns=COLUMNS) #df that will be written to a report file with info on model performance

value_or_zero = lambda x, dict: 0 if not x in dict else dict[x] #return dictionary value if key in dictionary, otherwise return 0
delete_punctuation = lambda x: ''.join([char for char in x if char not in string.punctuation]) #delete punctuation from string

#remove stopwords from text, put in lowercase, and delete punctuation
def clean(text):
    text = f' {text} '
    text = delete_punctuation(text.lower())
    for stopword in nltk.corpus.stopwords.words('english'):
        text = text.replace(f' {stopword} ', ' ')
    return text

#test single-variable output models,
#output of 1 means every prediction correct,
#0 means every prediction incorrect
def test(model, test_input, test_output):
    predictions = model.predict(test_input)
    wrong = [int(prediction == test_output[i]) for (i, prediction) in enumerate(predictions)]
    avg = sum(wrong) / len(wrong)
    return avg
    
#fit a model, save to file, and test
def fit_save_test(model, model_name, train_input, train_output, test_input, test_output):
    model.fit(train_input, train_output)
    filename = f'{MODEL_DIR}\\{model_name}.sav'
    pickle.dump(model, open(filename, 'wb'))
    avg = test(model, test_input, test_output)
    return avg
    
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
def train_representations(train_input, train_output, test_input, 
                          test_output, rep_method, algorithms, partition, task):
    global report_df
    
    rep_name = rep_method['name']
    rep_func = rep_method['function']
    rep_list = rep_method['wordlist']
    
    #represent text with that method
    temp_train_input = train_input.apply(lambda x: rep_func(x, rep_list)).values.tolist()
    temp_train_output = train_output.values.tolist()
    temp_test_input = test_input.apply(lambda x: rep_func(x, rep_list)).values.tolist()
    temp_test_output = test_output.values.tolist()
    
    for algorithm in algorithms:
        
        algo_name = algorithm['name']
        algo_model = algorithm['model']
    
        score = fit_save_test(algo_model, f'{algo_name}-{rep_name}-{task}', 
                              temp_train_input, temp_train_output,
                              temp_test_input, temp_test_output)
        
        print(f'Finished training {algo_name} model with {rep_name} representation, predicting {task}. Scored {score:.3f}.')
        temp_report = pd.DataFrame([algo_name, rep_name, task, score, partition], COLUMNS).transpose()
        report_df = pd.concat([report_df, temp_report])
    
#given a single partition, train and test each model with each rep method
def train_and_test(train_input, train_output, test_input, 
                   test_output, partition, task):
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
    
    rep_methods = [{'name': 'bow', 'function': bow, 'wordlist': wordlist}, #methods for representing text
                   {'name': 'freq', 'function': freq, 'wordlist': wordlist},
                   {'name': 'tfidf', 'function': tfidf, 'wordlist': word_dict}]
    
    algorithms = [{'name': 'decision-tree', 'model': tree.DecisionTreeClassifier()}, #algorithms for classification
                  {'name': 'svm', 'model': svm.SVC()},
                  {'name': 'naive-bayes', 'model': naive_bayes.GaussianNB()}]
    
    for rep_method in rep_methods:
        train_representations(train_input, train_output, test_input, test_output, 
                              rep_method, algorithms, partition, task)


#perform the entire process of training all models with all representations across all partitions,
#only for one task
def process_fit_test(df, output):
    length = len(df)
    jump = length // NUMBER_PARTITIONS #value increased each partition
    for partition in range(NUMBER_PARTITIONS):
        print(f'\nPARTITION NUMBER {partition+1}.\n')
    
        begin = partition * jump
        end = (partition+1) * jump
        
        test_df = df.iloc[begin:end]
        train_df = df.drop(test_df.index)

        train_input = train_df['text']
        train_output = train_df[output]
        test_input = test_df['text']
        test_output = test_df[output]
    
        train_and_test(train_input, train_output, test_input, test_output, 
                       partition, output)

def main():
    global report_df

    #Data preprocessing
    df = pd.read_csv(f'{BASIC}\\antisemitism_dataset.csv')
    df = df.sample(frac=1) #shuffle
    df['text'] = df['text'].apply(clean).astype(str)
    df = df[df['text'].apply(lambda x: len(x.split()) >= 1)]
    
    process_fit_test(df, 'classification')
    df = df[df['classification'] == 1]
    process_fit_test(df, 'type_of_antisemitism')
    
    report_df.to_csv('model_report.csv', index=False)

if __name__ == '__main__':
    main()