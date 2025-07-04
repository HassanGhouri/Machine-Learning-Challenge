from DecisionTree import DecisionTree, node_from_dict, node_to_dict
import numpy as np


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=self.n_features)

            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        return np.argmax(np.bincount(y))

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])

        return predictions


def forest_to_python_dict(forest):
    return {
        'n_trees': forest.n_trees,
        'max_depth': forest.max_depth,
        'min_samples_split': forest.min_samples_split,
        'n_features': forest.n_features,
        'trees': [
            {
                'root': node_to_dict(tree.root)
            }
            for tree in forest.trees
        ]
    }


def load_forest_from_python_dict(forest_data):
    rf = RandomForest(
        n_trees=forest_data['n_trees'],
        max_depth=forest_data['max_depth'],
        min_samples_split=forest_data['min_samples_split'],
        n_feature=forest_data['n_features']
    )
    rf.trees = []
    for tree_dict in forest_data['trees']:
        dt = DecisionTree(
            max_depth=rf.max_depth,
            min_samples_split=rf.min_samples_split,
            n_features=rf.n_features
        )
        dt.root = node_from_dict(tree_dict['root'])
        rf.trees.append(dt)
    return rf


def save_forest_to_file(forest, filename='forest_dict.txt'):
    d = forest_to_python_dict(forest)
    with open(filename, 'w') as f:
        f.write(repr(d))
