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
 * @file pybind_callback.hpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Callbacks for Python library.
 */

#pragma once

#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
#include "xcsf.h"
}

class EarlyStoppingCallback
{
  public:
    /**
     * @brief Constructor.
     * @param [in] kwargs Parameters and their values.
     *
     */
    EarlyStoppingCallback(py::str monitor, int patience, bool restore) :
        monitor(monitor), patience(patience), restore(restore)
    {
    }

    /**
     * @brief Checks whether early stopping criteria has been met.
     * @param [in] xcsf The XCSF data structure.
     * @param [in] metrics Dictionary of performance metrics.
     * @param [in] verbose Whether to print info.
     * @return whether early stopping criteria has been met.
     */
    bool
    should_stop(struct XCSF *xcsf, py::dict metrics, bool verbose)
    {
        if (!metrics.contains(monitor)) {
            std::ostringstream err;
            err << "invalid metric to monitor: " << monitor << std::endl;
            throw std::invalid_argument(err.str());
        }
        py::list data = metrics[monitor];
        py::list trials = metrics["trials"];
        const double current_error = py::cast<double>(data[data.size() - 1]);
        const int current_trial = py::cast<int>(trials[trials.size() - 1]);
        if (current_error < best_error) {
            best_error = current_error;
            best_trial = current_trial;
            if (restore) {
                xcsf_store_pset(xcsf);
                if (verbose) {
                    std::ostringstream status;
                    status << "checkpoint: ";
                    status << std::fixed << std::setprecision(5) << best_error;
                    status << " error at " << best_trial << " trials";
                    py::print(status.str());
                }
            }
        }
        if (current_trial - patience > best_trial) {
            if (restore) {
                xcsf_retrieve_pset(xcsf);
                if (verbose) {
                    std::ostringstream status;
                    status << "restoring system from trial " << best_trial;
                    status << " with error=";
                    status << std::fixed << std::setprecision(5) << best_error;
                    py::print(status.str());
                }
            }
            return true; // stop training
        }
        return false; // continue training
    }

  private:
    py::str monitor;
    int patience;
    bool restore;

    double best_error = std::numeric_limits<double>::max();
    int best_trial = 0;
};
