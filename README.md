# Decision Tree and Random Forest
Implementation of the classification decision tree and random forest supporting both `gini` and `entropy criteria`, and `max_depth`, 
`min_to_split` and `max_leaves` constraints.

Random forest is a collection of decision trees trained with dataset bagging and random feature subsampling.

## decision_tree
The decision tree algorithm is implemented in `decision_tree.py`. 

The file `decision_tree_main.py` shows how to use the implemented algorithm on some artificial data.

The example of the invocation of the program is:

`python decision_tree_main.py --criterion=gini --min_to_split=40 --max_leaves=4 --seed=92`

## random_forest
The random forest is implemented in `random_forest.py`.

The file `random_forest_main.py` shows how to use the implemented algorithm on some artificial data.

The example of the invocation of the program is:

`python random_forest.py --trees=3 --bootstrapping --feature_subsampling=0.5 --max_depth=2 --seed=46`


