#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from decision_tree import DecisionTree

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (i.e., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).
    decision_tree = DecisionTree()
    decision_tree.fit(train_data, train_target, criterion=args.criterion, max_depth=args.max_depth,
                      min_to_split=args.min_to_split, max_leaves=args.max_leaves)
    test_prediction = decision_tree.predict(test_data)
    train_prediction = decision_tree.predict(train_data)

    # Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_prediction)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_prediction)

    return train_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
