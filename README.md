# XCSF learning classifier system

An implementation of the XCSF [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) that can be built as a stand-alone binary or as a Python library. XCSF is an accuracy-based [online](https://en.wikipedia.org/wiki/Online_machine_learning) [evolutionary](https://en.wikipedia.org/wiki/Evolutionary_computation) [machine learning](https://en.wikipedia.org/wiki/Machine_learning) system with locally approximating functions that compute classifier payoff prediction directly from the input state. It can be seen as a generalisation of XCS where the prediction is a scalar value. XCSF attempts to find solutions that are accurate and maximally general over the global input space, similar to most machine learning techniques. However, it maintains the additional power to adaptively subdivide the input space into simpler local approximations.

See the project [wiki](https://github.com/rpreen/xcsf/wiki) for details on features, how to build, run, and use as a Python library.

*******************************************************************************

[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0)
[![Linux Build](https://img.shields.io/github/actions/workflow/status/rpreen/xcsf/ubuntu_build.yml?branch=master&logo=linux&logoColor=white&style=flat&label=Ubuntu)](https://github.com/rpreen/xcsf/actions?query=workflow%3A%22Ubuntu+build%22)
[![MacOS Build](https://img.shields.io/github/actions/workflow/status/rpreen/xcsf/macOS_build.yml?branch=master&logo=apple&logoColor=white&style=flat&label=macOS)](https://github.com/rpreen/xcsf/actions?query=workflow%3A%22macOS+build%22)
[![Windows Build](https://img.shields.io/appveyor/build/rpreen/xcsf?logo=windows&logoColor=white&style=flat&label=Windows)](https://ci.appveyor.com/project/rpreen/xcsf)
[![Latest Version](https://img.shields.io/github/v/release/rpreen/xcsf?style=flat)](https://github.com/rpreen/xcsf/releases)
[![DOI](https://zenodo.org/badge/28035841.svg)](https://zenodo.org/badge/latestdoi/28035841)

[![Codacy](https://img.shields.io/codacy/grade/2213b9ad4e034482bf058d4598d1618b?logo=codacy&style=flat)](https://www.codacy.com/gh/rpreen/xcsf/dashboard)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/rpreen/xcsf?logo=codefactor&style=flat)](https://www.codefactor.io/repository/github/rpreen/xcsf)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=rpreen_xcsf&metric=alert_status)](https://sonarcloud.io/dashboard?id=rpreen_xcsf)
[![codecov](https://codecov.io/gh/rpreen/xcsf/branch/master/graph/badge.svg?token=3bfaTvmJ8d)](https://codecov.io/gh/rpreen/xcsf)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=rpreen_xcsf&metric=ncloc)](https://sonarcloud.io/dashboard?id=rpreen_xcsf)

[![PyPI package](https://img.shields.io/pypi/v/xcsf.svg)](https://pypi.org/project/xcsf)
[![Python versions](https://img.shields.io/pypi/pyversions/xcsf.svg)](https://pypi.org/project/xcsf)
[![Downloads](https://static.pepy.tech/personalized-badge/xcsf?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/xcsf)
