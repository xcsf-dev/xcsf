/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file pybind_callback.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Interface for callbacks.
 */

#pragma once

#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
#include "xcsf.h"
}

class Callback
{
  public:
    virtual ~Callback() {}

    virtual bool
    run(struct XCSF *xcsf, py::dict metrics) = 0;

    virtual void
    finish(struct XCSF *xcsf) = 0;
};
