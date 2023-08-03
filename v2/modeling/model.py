from sklearn import tree
import pickle, time
import numpy as np

DATA_DIR = 'D:/data'
K = 10
MODEL = tree.DecisionTreeClassifier
METHODS = ['none', 'smote', 'over', 'under', 'aug']
MODELS = ['bert-base-uncased']
DATASETS = ['antisemitism_two', 'antisemitism_four', 'antisemitism_five', 'clothing_rating', 'clothing_topic', 'cyberbullying', 'disaster', 'website']

def read_files(data_dir: str, method_name: str, model_name: str, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Read the files associated with a particular model.'''
    with open(f'{DATA_DIR}/{method_name}/{model_name}/train-input-{dataset}', 'rb') as f:
        train_inputs = pickle.loads(f.read())
    with open(f'{DATA_DIR}/{method_name}/{model_name}/train-output-{dataset}', 'rb') as f:
        train_outputs = pickle.loads(f.read())
    with open(f'{DATA_DIR}/{method_name}/{model_name}/test-input-{dataset}', 'rb') as f:
        test_inputs = pickle.loads(f.read())

    return train_inputs, train_outputs, test_inputs

def model_partition(train_inputs: np.ndarray, train_outputs: np.ndarray, test_inputs: np.ndarray, 
                    step: int, train_jump: int, test_jump: int, num_classes: int) -> np.ndarray:
    '''Fit and test a particular partition within a dataset.'''
    train_inputs = train_inputs[step*train_jump:(step+1)*train_jump]
    train_outputs = train_outputs[step*train_jump:(step+1)*train_jump]
    test_inputs = test_inputs[step*test_jump:(step+1)*test_jump]

    #this is to deal with classes that are not present in a particular partition
    present_classes = list(set(train_outputs))
    replace_dict = {value: i for i, value in enumerate(present_classes)}
    reverse_dict = {i: value for i, value in enumerate(present_classes)}
    train_outputs = np.array([replace_dict[value] for value in train_outputs])

    model = MODEL().fit(train_inputs, train_outputs)
    return np.array([reverse_dict[value] for value in model.predict(test_inputs)])

def model_dataset(method_name: str, model_name: str, dataset: str) -> np.ndarray:
    '''Fully model a dataset with a particular resampling technique and give testing answers.'''
    train_inputs, train_outputs, test_inputs = read_files(DATA_DIR, method_name, model_name, dataset)
    train_jump, test_jump = len(train_inputs) // 10, len(test_inputs) // 10
    num_inputs = len(list(set(train_outputs)))
    return np.vstack([model_partition(train_inputs, train_outputs, test_inputs, step, train_jump, test_jump, num_inputs) for step in range(K)])

def write_to_file(predictions: np.ndarray, filename: str) -> None:
    '''Write the predictions of a model to a file'''
    with open(filename, 'wb') as f:
        pickle.dump(predictions, f)

def main():
    for method in METHODS:
        for model in MODELS:
            for dataset in DATASETS:
                start = time.time()
                write_to_file(model_dataset(method, model, dataset), f'{DATA_DIR}/{method}/{model}/predictions-{dataset}')
                end = time.time()
                print(f'Finished {method} method of {model} model on {dataset} dataset in {end-start:.2f} seconds.')

if __name__ == '__main__':
    main()