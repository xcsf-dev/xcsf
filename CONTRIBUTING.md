# Contributing

Contributions to this repository are very welcome. If you are interested in contributing feel free to contact me or create an issue in the [issue tracking system](https://github.com/rpreen/xcsf/issues). Alternatively, you may [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the project and submit a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork). All contributions must be made under the same license as the rest of the project: [GNU Public License v3](http://www.gnu.org/licenses/gpl-3.0). Ideally new code should be accompanied with appropriate test cases and documented.

*******************************************************************************

## Style guides

### C/C++

All C/C++ code should be formatted with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) using the provided style `.clang-format`.

For example: `clang-format -i -style=file */*.c */*.h */*.cpp`

Each data structure and function should be documented with minimal [Doxygen comments](https://www.doxygen.nl/manual/docblocks.html).

Please try to follow the style of existing code and use [const-correctness](https://en.wikipedia.org/wiki/Const_(computer_programming)).

### Python

All Python code should be linted with [pylint](https://www.pylint.org). A perfect score is not necessary, but try to clean up as much as is reasonable.

For example: `pylint python/example_rmux.py`

### CMake

All CMakeLists should be formatted with [cmake-format](https://github.com/cheshirekow/cmake_format) using the style `.cmake-format.yml`.

For example: `cmake-format -c .cmake-format.yml -i CMakeLists.txt`

### Yaml

All yaml files should be linted with [yamllint](https://github.com/adrienverge/yamllint).

For example: `yamllint .travis.yml`

## Version numbering

Version number format: `MAJOR.MINOR.BUILD` - specified in `xcsf.h` and `CMakeLists.txt` (PROJECT_VERSION).

* Increment `BUILD` number for minor bug fixes and changes that do not break compatability with file saving/loading.
* Increment `MINOR` number for major bug fixes and changes that break compatability with file saving/loading.
* Criteria for incrementing the `MAJOR` number to be determined later.
