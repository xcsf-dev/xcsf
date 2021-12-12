#!/usr/bin/python3
#
# Copyright (C) 2021 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

""" Python setup script for installing XCSF. """

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Creates a CMake extension module."""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Builds CMake extension."""

    def build_extension(self, ext):
        self.announce("Configuring CMake project", level=3)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
            "-DXCSF_MAIN=OFF",
            "-DXCSF_PYLIB=ON",
            "-DENABLE_DOXYGEN=OFF",
        ]
        build_args = [
            "--config",
            "Release",
        ]
        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=" + extdir]
            cmake_args += ["-GMinGW Makefiles"]
        else:
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir]
            build_args += ["-j4"]
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        self.announce("Building Python module", level=3)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="xcsf",
    version="1.1.4",
    license="GPL-3.0",
    maintainer="Richard Preen",
    maintainer_email="rpreen@gmail.com",
    description="XCSF learning classifier system: rule-based evolutionary machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpreen/xcsf",
    packages=find_packages(),
    package_data={"xcsf": ["__init__.pyi"]},
    ext_modules=[CMakeExtension("xcsf/xcsf")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    keywords="XCS, XCSF, learning-classifier-systems",
)
