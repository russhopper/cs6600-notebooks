import numpy as np
import pandas as pd

class DataSet:
    """
    A dataset for a machine learning problem. A dataset d has the following properties:
    d.examples   A list of examples. Each one contains both the features and the target.
    d.features   An array of the of feature names.
    d.target     An m by 1 array containing the values of y
    d.y          Same as d.target
    d.inputs     An n by m array containing the values of X
    d.X          Same as d.inputs
    d.N          Number of examples
    d.M          Number of dimensions
    d.name       The name of the data set (for output display only)
    
    """
    def __init__(self, data, features=None, y=None, name=None):
        """
        If y is True, the data contains the target as the last column
        If y is None or False, No target is available
        Else y is an array to be added as the last column of the examples  dataframe
        """
        self.__name = name
        if isinstance(data, pd.DataFrame):
            self.__examples = data
        else:
            self.__examples = pd.DataFrame(data, columns=features)
            
        if y is True:
            self.__examples.columns = [*self.__examples.columns[:-1], 'y']
        elif y is not False and y is not None:
            self.__examples['y'] = y
            
    
    @property
    def examples(self):
        return self.__examples
    
    @property
    def features(self):
        return self.__examples.columns[:-1].values
    
    @property
    def target(self):
        if 'y' in self.__examples.columns:
            return self.__examples['y'].values.reshape(self.N, 1)
        return None
    
    @property
    def y(self):
        return self.target
    
    @property
    def inputs(self):
        return self.__examples.iloc[:, :-1].values
    
    @property
    def X(self):
        return self.inputs
    
    @property
    def name(self):
        return self.__name
    
    @property
    def N(self):
        return self.__examples.shape[0]
    
    @property
    def M(self):
        return self.inputs.shape[1]
    
    def shuffled(self, random_state=None):
        rgen = np.random.RandomState(random_state)
        indexes = np.arange(self.N)
        rgen.shuffle(indexes)
        return DataSet(self.__examples.iloc[indexes])
    
    def train_test_split(self,start=0, end=None, test_portion=None, shuffle=False, random_state=None):
        """
        Splits the dataset into a training set and atest set. 
        If test_portion is specified, return that portion of the dataset as test 
        and the rest as training. 
        Otherwise, return the examples between start and end as test and the 
        rest as training.
        """
        indexes = np.arange(self.N)
        if shuffle is True:
            rgen = np.random.RandomState(random_state)
            rgen.shuffle(indexes)

        if test_portion is None:
            end = end or self.N
        else:
            if not isinstance(test_portion, float) or test_portion < 0 or test_portion > 1:
                raise TypeError("Only fractions between ]0,1[ are allowed")

            start = self.N - int(self.N * test_portion)
            end = self.N

        test = DataSet(self.examples.iloc[indexes[range(start, end)]])
        train = DataSet(pd.concat([self.examples.iloc[indexes[range(start)]], 
                                      self.examples.iloc[indexes[range(end, self.N)]]], axis=0))    
        return train, test

    def train_validation_test_split(self, validation_portion=.25, test_portion=.25, shuffle=False, random_state=None):
        '''
        Splits the dataset into training, validation, and testing sets.
        The default ratios are 50% training, 25% validation, and 25% testing
        '''
        indexes = np.arange(self.N)
        
        if shuffle:
            rgen = np.random.RandomState(random_state)
            rgen.shuffle(indexes)
        
        if not all(p<1 and p>0 and isinstance(p, float) for p in [validation_portion, test_portion]):
            raise TypeError("Only fractions between ]0,1[ are allowed")
        
        if validation_portion + test_portion >= 1:
            # they cannot exceed one for obvious reasons but they also can't equal one or there will be no test data
            raise Exception("validation_portion and test_portion must equal a ratio not equal to or exceeding 1.")
        
        valStart, testStart = int(self.N * (1 - (validation_portion + test_portion))), int(self.N * (1 - test_portion))
        
        train = DataSet(self.examples.iloc[indexes[range(0, valStart)]])
        validation = DataSet(self.examples.iloc[indexes[range(valStart, testStart)]])
        test = DataSet(self.examples.iloc[indexes[range(testStart,self.N)]])
        
        return train, validation, test
    
    def __repr__(self):
        return repr(self.examples)