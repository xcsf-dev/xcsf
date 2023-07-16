import json

from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import xcsf

RANDOM_STATE: int = 1

##########################################################
# Load some test data
##########################################################

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

##########################################################
# Initialise XCSF
##########################################################

xcs = xcsf.XCS(
    x_dim=4,
    y_dim=1,
    n_actions=1,
    omp_num_threads=12,
    random_state=RANDOM_STATE,
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

print(json.dumps(xcs.internal_params(), indent=4))

##########################################################
# Pipeline
##########################################################

model = make_pipeline(
    MinMaxScaler(feature_range=(-1.0, 1.0)),
    TransformedTargetRegressor(regressor=xcs, transformer=StandardScaler()),
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"train={train_score}")
print(f"test={test_score}")

##########################################################
# Grid search
##########################################################

# General parameters can be searched in the usual way:

parameters = {"beta": [0.1, 0.5]}

grid_search = GridSearchCV(xcs, parameters, scoring="neg_mean_squared_error")

grid_search.fit(X_train, y_train)

results = grid_search.cv_results_

for mean_score, std_score, params in zip(
    results["mean_test_score"], results["std_test_score"], results["params"]
):
    print("Mean Score:", -mean_score)
    print("Standard Deviation:", std_score)
    print("Parameters:", params)
    print("------------------------")

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# EA parameters require specifying a dict, but individual values
# can still be set because the other values are still initialised
# to their default values.

parameters = {"ea": [{"lambda": 2}, {"lambda": 10}, {"lambda": 50}]}

# However, for actions, conditions, and predictions, the WHOLE
# dict must be specified for each value to try in the search. This
# is because of the way XCSF uses kwargs to initialise values and they
# are reset each time. XCSF has so many different parameters that it
# is unfortunately necessary to do it this way.

parameters = {}
