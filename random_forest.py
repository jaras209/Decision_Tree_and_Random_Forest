import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, trees: int, random_generator=None, random_seed=42):
        self.random_generator = np.random.RandomState(random_seed) if not random_generator else random_generator
        self.decision_trees = [DecisionTree(self.random_generator) for _ in range(trees)]

    def fit(self, train_data, train_target, criterion='entropy', max_depth=None, min_to_split=2, max_leaves=None,
            feature_subsampling=1, bootstrapping=False):
        for tree in self.decision_trees:
            if bootstrapping:
                indices = self.random_generator.choice(len(train_data), size=len(train_data))
                data = train_data[indices]
                target = train_target[indices]

            else:
                data = train_data
                target = train_target

            tree.fit(data, target, criterion=criterion, max_depth=max_depth, min_to_split=min_to_split,
                     max_leaves=max_leaves, feature_subsampling=feature_subsampling)

    def predict(self, test_data):
        prediction = np.zeros(shape=(len(self.decision_trees), test_data.shape[0]), dtype=np.int8)
        for i, tree in enumerate(self.decision_trees):
            prediction[i] = tree.predict(test_data)

        prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=prediction)
        return prediction
