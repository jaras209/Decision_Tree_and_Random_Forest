import numpy as np
import heapq


class DecisionTree:
    class Node:
        def __init__(self, train_data, train_target, classes, criterion, random_generator=None, depth=0,
                     random_seed=42):
            self.train_data = train_data
            self.train_target = train_target
            self.size = train_data.shape[0]
            self.predicted_probability, _ = np.histogram(train_target, bins=np.arange(classes + 2), density=True)
            self.counts, _ = np.histogram(train_target, bins=np.arange(classes + 2), density=False)
            self.predicted_class = np.argmax(self.predicted_probability)
            self.classes = classes
            self.depth = depth
            self.split_feature = None
            self.split_point = None
            self.left = None
            self.right = None
            self.criterion = criterion
            self.random_generator = np.random.RandomState(random_seed) if not random_generator else random_generator

        def __lt__(self, other):
            return True

        def _gini_index_criterion(self):
            return self.size * np.sum(self.predicted_probability * (1 - self.predicted_probability))

        def _entropy_criterion(self):
            return - self.size * np.nansum(self.predicted_probability * np.log(self.predicted_probability))

        def criterion_value(self):
            if self.criterion == 'gini':
                return self._gini_index_criterion()

            else:
                return self._entropy_criterion()

        def _split(self, feature_subsampling):
            min_criterion_difference = np.inf
            this_criterion = self.criterion_value()
            best_left = None
            best_right = None
            feature_mask = self.random_generator.uniform(size=self.train_data.shape[1]) <= feature_subsampling

            for feature in np.arange(self.train_data.shape[1])[feature_mask]:
                unique = np.unique(self.train_data[:, feature])
                split_points = 0.5 * (unique[1:] + unique[:-1])
                for split_point in split_points:
                    left_mask = self.train_data[:, feature] <= split_point
                    right_mask = ~ left_mask
                    left = DecisionTree.Node(self.train_data[left_mask], self.train_target[left_mask], self.classes,
                                             self.criterion, random_generator=self.random_generator,
                                             depth=self.depth + 1)
                    right = DecisionTree.Node(self.train_data[right_mask], self.train_target[right_mask], self.classes,
                                              self.criterion, random_generator=self.random_generator,
                                              depth=self.depth + 1)
                    left_criterion = left.criterion_value()
                    right_criterion = right.criterion_value()

                    criterion_difference = left_criterion + right_criterion - this_criterion
                    if criterion_difference < min_criterion_difference:
                        min_criterion_difference = criterion_difference
                        best_left = left
                        best_right = right
                        self.split_feature = feature
                        self.split_point = split_point

            return best_left, best_right, min_criterion_difference

        def fit(self, max_depth, min_to_split, feature_subsampling):
            if self.size >= min_to_split and self.criterion_value() > 0:
                if max_depth and self.depth >= max_depth:
                    return None

                return self._split(feature_subsampling)

        def set_children(self, left, right):
            self.left = left
            self.right = right

        def predict(self, test_data):
            prediction = np.zeros(shape=test_data.shape[0], dtype=np.int8)
            if not self.left:
                prediction += self.predicted_class

            else:
                left_mask = test_data[:, self.split_feature] <= self.split_point
                right_mask = ~ left_mask
                prediction[left_mask] = self.left.predict(test_data[left_mask])
                prediction[right_mask] = self.right.predict(test_data[right_mask])

            return prediction

        def print(self):
            print("Split feature =", self.split_feature)
            print("Split point =", self.split_point)
            print("Size =", self.size)
            print("Criterion =", self.criterion)
            print("Criterion value =", self.criterion_value())
            print("Counts =", self.counts)
            print("Probabilities =", self.predicted_probability)
            if self.left:
                self.left.print()

            if self.right:
                self.right.print()

    def __init__(self, random_generator=None, random_seed=42):
        self.root = None
        self.random_generator = np.random.RandomState(random_seed) if not random_generator else random_generator

    def fit(self, train_data, train_target, criterion='gini', max_depth=None, min_to_split=2, max_leaves=None,
            feature_subsampling=1):

        self.root = self.Node(train_data, train_target, classes=np.max(train_target), criterion=criterion,
                              random_generator=self.random_generator)

        assert criterion == 'gini' or criterion == 'entropy', "criterion should be 'gini' or 'entropy'!"

        if max_leaves:
            leaves_priority_queue = []
            current_leaves = 1
            result = self.root.fit(max_depth=max_depth, min_to_split=min_to_split,
                                   feature_subsampling=feature_subsampling)
            if result:
                left, right, priority = result
                heapq.heappush(leaves_priority_queue, (priority, self.root, left, right))

            while leaves_priority_queue:
                priority, node, left, right = heapq.heappop(leaves_priority_queue)
                node.set_children(left, right)
                current_leaves += 1
                if current_leaves >= max_leaves:
                    break

                result = node.left.fit(max_depth=max_depth, min_to_split=min_to_split,
                                       feature_subsampling=feature_subsampling)
                if result:
                    left, right, priority = result
                    heapq.heappush(leaves_priority_queue, (priority, node.left, left, right))

                result = node.right.fit(max_depth=max_depth, min_to_split=min_to_split,
                                        feature_subsampling=feature_subsampling)
                if result:
                    left, right, priority = result
                    heapq.heappush(leaves_priority_queue, (priority, node.right, left, right))

        else:
            leaves_stack = []
            result = self.root.fit(max_depth=max_depth, min_to_split=min_to_split,
                                   feature_subsampling=feature_subsampling)
            if result:
                left, right, _ = result
                leaves_stack.append((self.root, left, right))

            while leaves_stack:
                node, left, right = leaves_stack.pop()
                node.set_children(left, right)

                result = node.left.fit(max_depth=max_depth, min_to_split=min_to_split,
                                       feature_subsampling=feature_subsampling)
                if result:
                    left, right, priority = result
                    leaves_stack.append((node.left, left, right))

                result = node.right.fit(max_depth=max_depth, min_to_split=min_to_split,
                                        feature_subsampling=feature_subsampling)
                if result:
                    left, right, priority = result
                    leaves_stack.append((node.right, left, right))

    def predict(self, test_data):
        return self.root.predict(test_data)

    def print(self):
        self.root.print()
