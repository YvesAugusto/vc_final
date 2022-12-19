import pickle
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument('-f', '--folder', type=str)
parser.add_argument('-n', '--name', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.name + "_test.pickle", "rb") as file:
        test_data = pickle.load(file)

    with open(args.name + "_train.pickle", "rb") as file:
        train_data = pickle.load(file)

    tree = RandomForestClassifier()
    train_data['features'] = np.array(train_data['features'])
    shape = train_data['features'].shape
    train_data['features'] = train_data['features'].reshape((shape[0], shape[1] * shape[2]))

    test_data['features'] = np.array(test_data['features'])
    shape = test_data['features'].shape
    test_data['features'] = test_data['features'].reshape((shape[0], shape[1] * shape[2]))

    pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()), 
            ('decision_tree', RandomForestClassifier()),
        ]
    )

    parameters = {
    'decision_tree__n_estimators': [40],
    }

    search = GridSearchCV(pipeline, parameters, cv=5)
    search.fit(train_data['features'], train_data['target'])
    best_params = {}
    for key, value in search.best_params_.items():
        key_ = key.split("__")[-1]
        best_params[key_] = value

    scoring = {
        'f1': make_scorer(f1_score),
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'accuracy': make_scorer(accuracy_score)
    }

    decision_tree = RandomForestClassifier(**best_params)
    print(best_params)

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    data = np.array(list(train_data['features']) + list(test_data['features']))
    target = np.array(list(train_data['target']) + list(test_data['target']))
    cv_results = cross_validate(decision_tree, data, target, cv=cv,
                                scoring=scoring)

    print(
        cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std(),
        cv_results['test_recall'].mean(), cv_results['test_recall'].std(),
        cv_results['test_precision'].mean(), cv_results['test_precision'].std(),
        cv_results['test_f1'].mean(), cv_results['test_f1'].std()
    )