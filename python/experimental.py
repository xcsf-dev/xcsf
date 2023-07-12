"""Experimental stuff."""
from sklearn import svm
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import xcsf

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xcsf.XCS(
    omp_num_threads=12,
    random_state=1,
    pop_init=True,
    max_trials=20000,
    perf_trials=5000,
    pop_size=500,
    loss_func="mse",
    set_subsumption=False,
    theta_sub=100,
    e0=0.005,
    alpha=1,
    nu=20,
    beta=0.1,
    delta=0.1,
    theta_del=50,
    init_fitness=0.01,
    init_error=0,
    m_probation=10000,
    stateful=True,
    compaction=False,
    ea={
        "select_type": "roulette",
        "theta_ea": 50,
        "lambda": 2,
        "p_crossover": 0.8,
        "err_reduc": 1,
        "fit_reduc": 0.1,
        "subsumption": False,
        "pred_reset": False,
    },
    condition={
        "type": "tree_gp",
        "args": {
            "min_constant": 0,
            "max_constant": 1,
            "n_constants": 100,
            "init_depth": 5,
            "max_len": 10000,
        },
    },
    prediction={
        "type": "neural",
        "args": {
            "layer_0": {
                "type": "connected",
                "activation": "relu",
                "n_init": 10,
                "evolve_weights": True,
                "evolve_functions": False,
                "evolve_connect": True,
                "evolve_neurons": False,
                "sgd_weights": True,
                "eta": 0.1,
                "evolve_eta": True,
                "eta_min": 1e-06,
                "momentum": 0.9,
                "decay": 0,
            },
            "layer_1": {
                "type": "connected",
                "activation": "softplus",
                "n_init": 1,
                "evolve_weights": True,
                "evolve_functions": False,
                "evolve_connect": True,
                "evolve_neurons": False,
                "sgd_weights": True,
                "eta": 0.1,
                "evolve_eta": True,
                "eta_min": 1e-06,
                "momentum": 0.9,
                "decay": 0,
            },
        },
    },
)

if False:
    # model fitting and scoring
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{train_score} :: {test_score}")

if False:
    # cross validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = cv_scores.mean()
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", mean_cv_score)

if False:
    # pipeline
    model = make_pipeline(
        MinMaxScaler(feature_range=(-1.0, 1.0)),
        TransformedTargetRegressor(regressor=model, transformer=StandardScaler()),
    )
    model.fit(X, y)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{train_score} :: {test_score}")

if False:
    # grid search SVM
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    grid_search = GridSearchCV(svc, parameters)
    grid_search.fit(X, y)
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

if True:
    # grid search XCSF
    # parameters = {"beta": [0.1, 0.5]}
    parameters = {"ea": [{"lambda": 2}, {"lambda": 10}, {"lambda": 50}]}
    grid_search = GridSearchCV(model, parameters)  # 5 folds as default
    grid_search.fit(X, y)
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
