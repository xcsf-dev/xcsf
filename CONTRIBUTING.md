# Contributing

Contributions to this repository are very welcome. If you are interested in contributing feel free to contact me or create an issue in the [issue tracking system](https://github.com/rpreen/xcsf/issues). Alternatively, you may [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the project and submit a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). All contributions must be made under the same license as the rest of the project: [GNU Public License v3](http://www.gnu.org/licenses/gpl-3.0).

*******************************************************************************

## Style guides

### C/C++

All C/C++ code should be formatted with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) using the provided style `.clang-format`.

For example: `clang-format -i -style=file */*.c */*.h */*.cpp`

Each data structure and function should be documented with minimal [Doxygen comments](https://www.doxygen.nl/manual/docblocks.html). Try to follow the style of existing code.

### Python

All Python code should be linted with [pylint](https://www.pylint.org). A perfect score is not necessary, but try to clean up as much as is reasonable.

For example: `pylint python/example_rmux.py`

### CMake

All CMakeLists should be linted and formatted using the style `.cmake-format.yml`. Linting is performed with [cmake-lint](https://github.com/cmake-lint/cmake-lint) and formatting with [cmake-format](https://github.com/cheshirekow/cmake_format).

For example:

`cmake-lint -c .cmake-format.yml CMakeLists.txt`

`cmake-format -c .cmake-format.yml -i CMakeLists.txt`

### Yaml

All yaml files should be linted with [yamllint](https://github.com/adrienverge/yamllint).

For example: `yamllint .travis.yml`
