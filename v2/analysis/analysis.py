#Imports
from matplotlib import pyplot as plt
from sklearn import metrics
import pickle, typing
import numpy as np
import pandas as pd

#Constants and global variables
DATA_DIR = 'D:/data'
K = 10
METHODS = ['none', 'cnn', 'border', 'smote', 'over', 'under', 'aug', 'aug_fine']
CLASSIFIERS = ['dtree']
MODELS = ['bert-base-uncased',]
DATASETS = ['antisemitism_two', 
            'antisemitism_four', 
            'antisemitism_five',  
            'disaster', 
            'website', 
            'clothing_rating', 
            'clothing_topic', 
            'cyberbullying',
            'news_mild_multi',
            'news_mild_two',
            'news_multi',
            'news_semi_multi',
            'news_severe_multi',
            'news_severe_two',
            ]
METRICS = [metrics.f1_score,] #metrics.balanced_accuracy_score]

#Analysis helper functions
def open_analyze_file(method: str, model: str, dataset: str, metric: typing.Callable, classifier: str) -> float:
    '''Open and analyze a prediction and actual results file.'''
    with open(f'{DATA_DIR}/{method}/{model}/predictions-{dataset}-{classifier}', 'rb') as f:
        predictions = np.array(pickle.loads(f.read()).ravel())
    with open(f'{DATA_DIR}/{method}/{model}/test-output-{dataset}', 'rb') as f:
        actual = np.array(pickle.loads(f.read()))
        actual = actual[:-(actual.shape[0] % K)] if actual.shape[0] != predictions.shape[0] else actual
    
    return metric(actual, predictions, average='macro') if metric == METRICS[0] else metric(actual, predictions)
    
#High-level analysis functions
def analyze_method(method_name: str, model_name: str, metric: typing.Callable, classifier: str) -> np.ndarray:
    '''Analyze a method with a particular model and metric, giving scores on all datasets.'''
    return np.array([open_analyze_file(method_name, model_name, dataset, metric, classifier) for dataset in DATASETS])

def analyze_methods(model_name: str, metric: typing.Callable, classifier: str) -> np.ndarray:
    '''Analyze all methods with a particular model and metric.'''
    return np.array([analyze_method(method, model_name, metric, classifier) for method in METHODS])

def main():
    for model in MODELS:
        print(f'\n{model}')
        for metric in METRICS:
            for classifier in CLASSIFIERS:
                print(classifier)
                df = pd.DataFrame(data=analyze_methods(model, metric, classifier), columns=DATASETS, index=METHODS)
                df.loc['highest_score'] = df.apply(max, axis=0)
                df.loc['highest'] = df.apply('idxmax', axis=0)
                print(metric)
                print(df)
                df.to_csv(f'output/{model}-{classifier}.csv')
                print('\n\n')

if __name__ == '__main__':
    main()