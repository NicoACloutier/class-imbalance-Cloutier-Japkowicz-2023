import numpy as np
import collections

#a modified knn that multiplies the distances of the input point to datapoints by certain values depending on class disparities.
#only to be used with classification.

class ModifiedKNN:
    def __init__(inputs: np.ndarray, outputs: np.ndarray, k: int, classes: int, modifiers: np.ndarray = None) -> None:
        '''
        inputs: np.ndarray. Two dimensional array where each row is observation and each column is input dimension
        outputs: np.ndarray. Vector where each value is a scalar classification for an observation, aligned to `inputs`'s rows
        k: int. The number k neighbors looked at.
        classes: int. The number of classes.
        modifiers: np.ndarray. Only specify if you don't want the default way of determining the modifiers, will override.
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.k = k
        self.classes = classes
        if modifiers:
            self.modifiers = modifiers
        else:
            self.modifiers = get_modifiers(self.outputs)
        self.output_modifiers = [self.modifiers[item] for item in self.outputs]

    #get the modification scalars from the outputs
    def get_modifiers(outputs: np.ndarray) -> np.ndarray:
        counter = collections.Counter(outputs)
        modifiers = np.array([counter[i] for i in range(self.classes)]) / np.array([len(outputs)] * self.classes)
        return modifiers

    def classify(input_vector: np.ndarray) -> int:
        '''
        Classify on a single input observation.
        input_vector: np.ndarray. Aligned in dimensionality to the input data.
        '''
        temp_inputs = np.copy(self.inputs)
        temp_inputs -= input_vector
        temp_inputs = np.sqrt(temp_inputs.dot(temp_inputs))
        temp_inputs /= self.output_modifiers
        
        indeces = np.argpartition(temp_inputs, self.k)[:k]
        answers = np.array([outputs[index] for index in indeces])
        values, counts = np.unique(answers, return_counts=True)
        mode_index = np.argmax(counts)
        return values[mode_index]

