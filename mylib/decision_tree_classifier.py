# Version 1.0
import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    """ A basic ID3 Decision Tree"""

    def __init__(self, dataset, nFeatures=0):
        self.dataset = dataset
        self.tree = pd.Series(dtype=object)
        self.nFeatures = nFeatures
        self.make_tree(self.dataset.examples, self.tree)
    
    def entropy(self, df):
        p = df.iloc[:, -1].value_counts() / len(df)
        return (-p * np.log2(p)).sum()

    def info_gain(self, df, feature):
        p = df[feature].value_counts() / len(df)

        for v in p.index:
            p.loc[v] *= self.entropy(df[df[feature] == v])

        return self.entropy(df) - p.sum()

    def best_feature(self, df):
        features = df.columns[:-1].copy().values
        if self.nFeatures != 0:
            f_indexes = np.arange(min(self.nFeatures, len(features)))
            np.random.shuffle(f_indexes)
            features = features[f_indexes]
        
        info = pd.DataFrame({"feature": features})
        info['gain'] = [self.info_gain(df, f) for f in features]
        return info['feature'][info['gain'].argmax()]
    

    def print_tree(self, name='', node=None, depth=1):
        if node is None:
            node = self.tree
            
        for f in node.index:
            if isinstance(node[f], tuple):
                if f != '-^-':
                    print(' ' * depth, f, ' => ', node[f], sep='')
            else:
                print(' ' * depth, f, ': ', sep='')
                self.print_tree(f, node[f], depth + 1)
                
    def make_tree(self, df, node, feature=None):
        if feature is None:
            feature = self.best_feature(df)
        
        node[feature] = pd.Series(dtype=object)
        
        # Store the plurality vote class at the feature level
        # under a "hidden" _^_ key just in case we need it for
        # when the unseen example does not lead to a leaf.
        node[feature]['-^-'] = (feature, df.iloc[:, -1].mode()[0])
        
        fvalues = df[feature].unique()
        for v in fvalues:
            d = df[df[feature] == v]
            n_classes = len(d.iloc[:, -1].unique())
            if n_classes == 1:
                node[feature][v] = ('L', d.iloc[:, -1].iloc[0])
            elif n_classes > 1:
                d = d.drop([feature], axis=1)
                if len(d.columns) == 1: 
                    node[feature][v] = ('L', d.iloc[:, -1].mode()[0])
                else:
                    next_best_feature = self.best_feature(d)
                    node[feature][v] = pd.Series(dtype=object)
                    self.make_tree(d, node[feature][v] ,next_best_feature)
            else:
                pass

    def predict(self, unseen, node=None):
        """
        Returns the most probable label (or class) for each unseen input. The
        unseen needs to be a data series with the same features (as indexes) as the 
        training data. It can also be a data frame with the same features as 
        the training data.
        """
        if unseen.ndim == 1:
            if node is None:
                node = self.tree

            feature = node.index[0]
            children = node[feature]
            value = unseen[feature]
            for c in children.index:
                if c == value:
                    if isinstance(children[c], tuple):
                        return children[c][1]
                    else:
                        return self.predict(unseen, children[c])
                    
            # If this is reached, then a leaf was not reached. We return
            # a plurality vote at the deepest node reached.
            return children['-^-'][1]
        else:
            return np.array([self.predict(unseen.iloc[i,:]) for i in range(len(unseen))])  
    
    def __repr__(self):
        return repr(self.tree)
  
