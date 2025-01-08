import numpy as np
from collections import Counter
from joblib import Parallel, delayed


class Node:
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=int))
        percentages = counts / len(s)
        entropy = -np.sum(p * np.log2(p) for p in percentages if p > 0)
        return entropy

    def _information_gain(self, parent, left_child, right_child):
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

    def _best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape

        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            unique_vals = np.unique(X_curr)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2  # Midpoints as thresholds

            for threshold in thresholds:
                mask = X_curr <= threshold
                y_left, y_right = y[mask], y[~mask]

                if len(y_left) > 0 and len(y_right) > 0:
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'mask_left': mask,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split

    def _build(self, X, y, depth=0):
        if len(y) >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best.get('gain', 0) > 0:
                left = self._build(X[best['mask_left']], y[best['mask_left']], depth + 1)
                right = self._build(X[~best['mask_left']], y[~best['mask_left']], depth + 1)
                return Node(feature=best['feature_index'], threshold=best['threshold'],
                            data_left=left, data_right=right, gain=best['gain'])
        return Node(value=Counter(y).most_common(1)[0][0])

    def fit(self, X, y):
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        if tree.value is not None:
            return tree.value
        if x[tree.feature] <= tree.threshold:
            return self._predict(x, tree.data_left)
        return self._predict(x, tree.data_right)

    def predict(self, X):
        return [self._predict(x, self.root) for x in X]


class RandomForest_manual:
    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []

    @staticmethod
    def _sample(X, y):
        np.random.seed(42)
        n_rows = X.shape[0]
        indices = np.random.choice(n_rows, size=n_rows, replace=True)
        X_sample, y_sample = X[indices], y[indices]
        
        # Menghapus dimensi channel tambahan (C=1)
        X_sample = np.squeeze(X_sample)  # Menghapus dimensi 1
        print(f"Sampled X shape after squeeze: {X_sample.shape}")  # Debugging
        print(f"Sampled y shape: {y_sample.shape}")  # Debugging
        return X_sample, y_sample




    def fit_tree(self, X, y):
        print(f"Fitting tree with X shape: {X.shape}, y shape: {y.shape}")  # Debugging
        tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        tree.fit(X, y)
        return tree


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Menghapus dimensi ekstra jika ada
        X = np.squeeze(X)  # Pastikan bentuknya adalah (n_samples, n_features)
        
        print(f"Shape of X before training: {X.shape}")  # Debugging
        print(f"Shape of y: {y.shape}")  # Debugging
        
        self.decision_trees = Parallel(n_jobs=-1)(
            delayed(self.fit_tree)(*self._sample(X, y)) for _ in range(self.num_trees)
        )


    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        predictions = [Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return predictions
