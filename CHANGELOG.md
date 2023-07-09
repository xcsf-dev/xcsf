# Changelog

## Version 1.2.8 (Jul 9, 2023)

Changes:
*    Add support for pickle to Python library ([#72](https://github.com/rpreen/xcsf/pull/72))

## Version 1.2.7 (May 1, 2023)

Changes:
*    Add defaults to Python API calls to allow a cleaner syntax.
*    Fix an input check on the cover parameter in predict/score.
*    Update Python cartpole example to latest version and refresh other examples.

## Version 1.2.6 (Apr 23, 2023)

Changes:
*    Add an optional argument to the Python predict/score functions that specifies the value to return for a sample if the match set is empty instead of invoking covering.

## Version 1.2.5 (Oct 3, 2022)

Changes:
*    Add docstrings and variable names to Python library.
*    Fix minor memory leak when printing parameters.

## Version 1.2.4 (Oct 1, 2022)

Changes:
*    Python library throws exceptions rather than hard exiting where possible.
*    Fix minor memory leak when printing parameters.

## Version 1.2.3 (Sep 23, 2022)

Changes:
*    Fix numpy subnormal warning.

## Version 1.2.2 (Sep 20, 2022)

Changes:
*    Best action selection now breaks ties randomly.

## Version 1.2.1 (Sep 3, 2022)

Changes:
*    Fix hyperrectangle_ubr json population seeding import/export asymmetry.

## Version 1.2.0 (Sep 3, 2022)

Changes:
*    Added extra JSON parsing input checks.
*    Cleaned up Python examples.
*    Added Jupyter notebook examples.
*    Renamed hyperrectangle conditions with hyperrectangle_csr.
*    Added unordered-bound hyperrectangle conditions with hyperrectangle_ubr.

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
*    Fixed Python read/write `EA_SELECT_TYPE` asymmetry.
*    Cleaned up Python interface and added input checks.
*    Moved ini config to a subdirectory.
*    Main executable building now optional.
*    Added pypi package building and support building Windows Python 3.10.
*    Updated libraries to latest versions.

## Version 1.1.2 (Oct 31, 2021)

Changes:
*    Clean up - move to wiki.
*    Addition of set seed function.
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
