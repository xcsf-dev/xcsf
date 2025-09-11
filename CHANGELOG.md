# Changelog

## Version 1.4.9 (Sep 11, 2025)

Changes:
*   Fix constant prediction early stopping rollback ([#214](https://github.com/xcsf-dev/xcsf/pull/214))

## Version 1.4.8 (Aug 26, 2025)

Changes:
*   Add checks for array contiguity ([#199](https://github.com/xcsf-dev/xcsf/pull/199))
*   Add `clean` parameter to wipe existing population to `json_read` ([#202](https://github.com/xcsf-dev/xcsf/pull/202))
*   Add Ruff linting and formatting for Python ([#196](https://github.com/xcsf-dev/xcsf/pull/196))
*   Fix Python `json_read` crashing when the new population is empty ([#205](https://github.com/xcsf-dev/xcsf/pull/205))
*   Reduce `max_trials` in Python tests for speed ([#206](https://github.com/xcsf-dev/xcsf/pull/206))
*   Update Python packaging: move `setup.cfg` metadata to `pyproject.toml` ([#207](https://github.com/xcsf-dev/xcsf/pull/207))
*   Fix macOS building with AppleClang ([#210](https://github.com/xcsf-dev/xcsf/pull/210))
*   Add support for Python 3.14 ([#211](https://github.com/xcsf-dev/xcsf/pull/211))
*   Update pybind11 to v3.0.1 and doctest to v2.4.12 ([#212](https://github.com/xcsf-dev/xcsf/pull/212))

## Version 1.4.7 (Aug 19, 2024)

Changes:
*   Fix saving and loading DGP graphs from persistent storage ([#151](https://github.com/xcsf-dev/xcsf/pull/151))
*   Add Python tests ([#149](https://github.com/xcsf-dev/xcsf/pull/149))
*   Drop support for Python 3.8; Add support for Python 3.13 ([#155](https://github.com/xcsf-dev/xcsf/pull/155))

## Version 1.4.6 (Jul 21, 2024)

Changes:
*   Update Python packaging ([#124](https://github.com/xcsf-dev/xcsf/pull/124))
*   Add Python `validation_data` input check ([#136](https://github.com/xcsf-dev/xcsf/pull/136))
*   Fix sanitizer, free test memory, fix neural saving ([#146](https://github.com/xcsf-dev/xcsf/pull/146))
*   Update cJSON and pybind11 libs ([#147](https://github.com/xcsf-dev/xcsf/pull/147))

## Version 1.4.5 (Feb 23, 2024)

Changes:
*   Remove compilation with `-flto` ([#128](https://github.com/xcsf-dev/xcsf/pull/128))

## Version 1.4.4 (Jan 21, 2024)

Changes:
*   Improve parameter checks ([#112](https://github.com/xcsf-dev/xcsf/pull/112), [#114](https://github.com/xcsf-dev/xcsf/pull/114))
*   Improve hyperrectangle CSR semantics ([#117](https://github.com/xcsf-dev/xcsf/pull/117))
*   Fix a sampling bug in Python `fit()` when `shuffle=False` ([#122](https://github.com/xcsf-dev/xcsf/pull/122))

## Version 1.4.3 (Nov 27, 2023)

Changes:
*   Make predict function deterministic even with parallelism ([#109](https://github.com/xcsf-dev/xcsf/pull/109))

## Version 1.4.2 (Nov 10, 2023)

Changes:
*   Fix a bug serialising initial population set filename ([#104](https://github.com/xcsf-dev/xcsf/pull/104))
*   Add tests ([#105](https://github.com/xcsf-dev/xcsf/pull/105))

## Version 1.4.1 (Nov 3, 2023)

Changes:
*   Make `random_state` parameter sklearn compatible ([#97](https://github.com/xcsf-dev/xcsf/pull/97))

## Version 1.3.1 (Jul 28, 2023)

Changes:
*   Add tests.
*   Fix a bug serialising initial population set filename.
*   Fix a bug copying convolutional layer momentum.

## Version 1.3.0 (Jul 20, 2023)

Changes:
*   Accept 1-D flattened Python inputs in fit/score/predict ([#81](https://github.com/xcsf-dev/xcsf/pull/81))
*   Fix setting max trials via JSON ([#83](https://github.com/xcsf-dev/xcsf/pull/83))
*   Remove Python `seed()` and add `RANDOM_STATE` parameter for setting seed ([#86](https://github.com/xcsf-dev/xcsf/pull/86))
*   Major Python API update: sklearn compatibility ([#77](https://github.com/xcsf-dev/xcsf/pull/77))
*   Add Python callback support with EarlyStoppingCallback ([#77](https://github.com/xcsf-dev/xcsf/pull/77))
*   Add parameter to read initial population from JSON; also works in stand-alone binary now ([#77](https://github.com/xcsf-dev/xcsf/pull/77))
*   Add hyperparameter tuning example ([#77](https://github.com/xcsf-dev/xcsf/pull/77))
*   Add CheckpointCallback ([#88](https://github.com/xcsf-dev/xcsf/pull/88))

## Version 1.2.9 (Jul 9, 2023)

Changes:
*    Fix deserialization of tree-GP and DGP conditions ([#74](https://github.com/xcsf-dev/xcsf/pull/74))
*    Update libraries ([#75](https://github.com/xcsf-dev/xcsf/pull/75))

## Version 1.2.8 (Jul 9, 2023)

Changes:
*    Add support for pickle to Python library ([#72](https://github.com/xcsf-dev/xcsf/pull/72))

## Version 1.2.7 (May 1, 2023)

Changes:
*    Add defaults to Python API calls to allow a cleaner syntax ([#60](https://github.com/xcsf-dev/xcsf/pull/60))
*    Fix an input check on the cover parameter in predict/score ([#60](https://github.com/xcsf-dev/xcsf/pull/60))
*    Update Python cartpole example to latest version and refresh other examples ([#60](https://github.com/xcsf-dev/xcsf/pull/60))

## Version 1.2.6 (Apr 23, 2023)

Changes:
*    Add an optional argument to the Python predict/score functions that specifies the value to return for a sample if the match set is empty instead of invoking covering ([#59](https://github.com/xcsf-dev/xcsf/pull/59))

## Version 1.2.5 (Oct 3, 2022)

Changes:
*    Add docstrings and variable names to Python library ([#43](https://github.com/xcsf-dev/xcsf/pull/43))
*    Fix minor memory leak when printing parameters ([#44](https://github.com/xcsf-dev/xcsf/pull/44))

## Version 1.2.4 (Oct 1, 2022)

Changes:
*    Python library throws exceptions rather than hard exiting where possible ([#41](https://github.com/xcsf-dev/xcsf/pull/41), [#42](https://github.com/xcsf-dev/xcsf/pull/42))
*    Fix minor memory leak when printing parameters.

## Version 1.2.3 (Sep 23, 2022)

Changes:
*    Fix numpy subnormal warning.

## Version 1.2.2 (Sep 20, 2022)

Changes:
*    Best action selection now breaks ties randomly ([#39](https://github.com/xcsf-dev/xcsf/pull/39))

## Version 1.2.1 (Sep 3, 2022)

Changes:
*    Fix hyperrectangle_ubr json population seeding import/export asymmetry.

## Version 1.2.0 (Sep 3, 2022)

Changes:
*    Added extra JSON parsing input checks.
*    Cleaned up Python examples.
*    Added Jupyter notebook examples.
*    Renamed hyperrectangle conditions with hyperrectangle_csr ([#35](https://github.com/xcsf-dev/xcsf/pull/35))
*    Added unordered-bound hyperrectangle conditions with hyperrectangle_ubr ([#35](https://github.com/xcsf-dev/xcsf/pull/35))

## Version 1.1.6 (Dec 27, 2021)

Changes:
*    Fixed cross-platform compiling and wheel building.

## Version 1.1.5 (Dec 27, 2021)

Changes:
*    Added functions to import parameters as JSON.
*    Stand-alone binary config file now in JSON format.
*    Python library and stand-alone config parameter setters use same JSON import functions.
*    Added functions inserting classifiers into the population in JSON.
*    Minor refactoring.

## Version 1.1.4 (Dec 12, 2021)

Changes:
*    Python utilities included in packaging.
*    Python code formatted with black.
*    Added Python type hints and stub for mypy type checking.
*    Hyphens in variable names/types converted to underscore.

## Version 1.1.3 (Dec 4, 2021)

Changes:
*    Added cJSON library.
*    Added functions returning/printing classifiers in JSON.
*    Added functions returning/printing parameters in JSON.
*    Added Python classes for visualising tree and graph conditions.
*    Fixed Python read/write `EA_SELECT_TYPE` asymmetry ([#16](https://github.com/xcsf-dev/xcsf/pull/16))
*    Cleaned up Python interface and added input checks.
*    Moved ini config to a subdirectory.
*    Main executable building now optional.
*    Added pypi package building and support building Windows Python 3.10.
*    Updated libraries to latest versions.

## Version 1.1.2 (Oct 31, 2021)

Changes:
*    Clean up - move to wiki.
*    Addition of set seed function ([#14](https://github.com/xcsf-dev/xcsf/pull/14))
*    Minor bug fix to action/match set numerosity.
*    Neural layers print more detailed information.

## Version 1.1.1 (Dec 10, 2020)

Changes:
*    Fixed CMakeLists to remove schematic warning.

## Version 1.1.0 (Dec 10, 2020)

Changes:
*    Increased documentation.
*    Additional unit tests.
*    Float-to-binary changed to human readable ordering.
*    Fixed opening config ini file on Windows.
*    Minor source refactoring/cleaning.
*    Added neural layer initialisation parameters to file saving/loading.

## Version 1.0.0 (Nov 17, 2020)

First version.
