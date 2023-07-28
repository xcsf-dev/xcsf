# Changelog

## Version 1.3.1 (Jul 28, 2023)

Changes:
*   Add tests.
*   Fix a bug serialising initial population set filename.
*   Fix a bug copying convolutional layer momentum.

## Version 1.3.0 (Jul 20, 2023)

Changes:
*   Accept 1-D flattened Python inputs in fit/score/predict ([#81](https://github.com/rpreen/xcsf/pull/81))
*   Fix setting max trials via JSON ([#83](https://github.com/rpreen/xcsf/pull/83))
*   Remove Python `seed()` and add `RANDOM_STATE` parameter for setting seed ([#86](https://github.com/rpreen/xcsf/pull/86))
*   Major Python API update: sklearn compatibility ([#77](https://github.com/rpreen/xcsf/pull/77))
*   Add Python callback support with EarlyStoppingCallback ([#77](https://github.com/rpreen/xcsf/pull/77))
*   Add parameter to read initial population from JSON; also works in stand-alone binary now ([#77](https://github.com/rpreen/xcsf/pull/77))
*   Add hyperparameter tuning example ([#77](https://github.com/rpreen/xcsf/pull/77))
*   Add CheckpointCallback ([#88](https://github.com/rpreen/xcsf/pull/88))

## Version 1.2.9 (Jul 9, 2023)

Changes:
*    Fix deserialization of tree-GP and DGP conditions ([#74](https://github.com/rpreen/xcsf/pull/74))
*    Update libraries ([#75](https://github.com/rpreen/xcsf/pull/75))

## Version 1.2.8 (Jul 9, 2023)

Changes:
*    Add support for pickle to Python library ([#72](https://github.com/rpreen/xcsf/pull/72))

## Version 1.2.7 (May 1, 2023)

Changes:
*    Add defaults to Python API calls to allow a cleaner syntax ([#60](https://github.com/rpreen/xcsf/pull/60))
*    Fix an input check on the cover parameter in predict/score ([#60](https://github.com/rpreen/xcsf/pull/60))
*    Update Python cartpole example to latest version and refresh other examples ([#60](https://github.com/rpreen/xcsf/pull/60))

## Version 1.2.6 (Apr 23, 2023)

Changes:
*    Add an optional argument to the Python predict/score functions that specifies the value to return for a sample if the match set is empty instead of invoking covering ([#59](https://github.com/rpreen/xcsf/pull/59))

## Version 1.2.5 (Oct 3, 2022)

Changes:
*    Add docstrings and variable names to Python library ([#43](https://github.com/rpreen/xcsf/pull/43))
*    Fix minor memory leak when printing parameters ([#44](https://github.com/rpreen/xcsf/pull/44))

## Version 1.2.4 (Oct 1, 2022)

Changes:
*    Python library throws exceptions rather than hard exiting where possible ([#41](https://github.com/rpreen/xcsf/pull/41), [#42](https://github.com/rpreen/xcsf/pull/42))
*    Fix minor memory leak when printing parameters.

## Version 1.2.3 (Sep 23, 2022)

Changes:
*    Fix numpy subnormal warning.

## Version 1.2.2 (Sep 20, 2022)

Changes:
*    Best action selection now breaks ties randomly ([#39](https://github.com/rpreen/xcsf/pull/39))

## Version 1.2.1 (Sep 3, 2022)

Changes:
*    Fix hyperrectangle_ubr json population seeding import/export asymmetry.

## Version 1.2.0 (Sep 3, 2022)

Changes:
*    Added extra JSON parsing input checks.
*    Cleaned up Python examples.
*    Added Jupyter notebook examples.
*    Renamed hyperrectangle conditions with hyperrectangle_csr ([#35](https://github.com/rpreen/xcsf/pull/35))
*    Added unordered-bound hyperrectangle conditions with hyperrectangle_ubr ([#35](https://github.com/rpreen/xcsf/pull/35))

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
*    Fixed Python read/write `EA_SELECT_TYPE` asymmetry ([#16](https://github.com/rpreen/xcsf/pull/16))
*    Cleaned up Python interface and added input checks.
*    Moved ini config to a subdirectory.
*    Main executable building now optional.
*    Added pypi package building and support building Windows Python 3.10.
*    Updated libraries to latest versions.

## Version 1.1.2 (Oct 31, 2021)

Changes:
*    Clean up - move to wiki.
*    Addition of set seed function ([#14](https://github.com/rpreen/xcsf/pull/14))
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
