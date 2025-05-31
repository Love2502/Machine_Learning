import numpy as np
from collections import Counter

class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, features, labels):
        self.root = self.build_tree(features, labels, depth=0)

    def predict(self, features):
        return [self.predict_one(x, self.root) for x in features]

    def build_tree(self, features, labels, depth):
        if depth >= self.max_depth or len(set(labels)) == 1:
            return self.majority_label(labels)

        feature_index, split_value = self.best_split(features, labels)
        if feature_index is None:
            return self.majority_label(labels)

        left_mask = features[:, feature_index] <= split_value
        right_mask = features[:, feature_index] > split_value

        left_branch = self.build_tree(features[left_mask], labels[left_mask], depth + 1)
        right_branch = self.build_tree(features[right_mask], labels[right_mask], depth + 1)

        return (feature_index, split_value, left_branch, right_branch)

    def best_split(self, features, labels):
        best_score = -1
        best_feature = None
        best_value = None

        for col in range(features.shape[1]):
            values = np.unique(features[:, col])
            for val in values:
                left = labels[features[:, col] <= val]
                right = labels[features[:, col] > val]

                if len(left) == 0 or len(right) == 0:
                    continue

                score = abs(len(set(left)) - len(set(right)))
                if score > best_score:
                    best_score = score
                    best_feature = col
                    best_value = val

        return best_feature, best_value

    def majority_label(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature_index, split_value, left, right = node
        if x[feature_index] <= split_value:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)
