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
    EarlyStoppingCallback(py::str monitor, int patience, bool restore,
                          double min_delta, int start_from, bool verbose) :
        monitor(monitor),
        patience(patience),
        restore(restore),
        min_delta(min_delta),
        start_from(start_from),
        verbose(verbose)
    {
        std::ostringstream err;
        std::string str = monitor.cast<std::string>();
        if (str != "train" && str != "val") {
            err << "invalid metric to monitor: " << str << std::endl;
            throw std::invalid_argument(err.str());
        }
        if (patience < 0) {
            err << "patience must be greater than zero" << std::endl;
            throw std::invalid_argument(err.str());
        }
        if (min_delta < 0) {
            err << "min_delta must be greater than zero" << std::endl;
            throw std::invalid_argument(err.str());
        }
    }

    /**
     * @brief Checkpoints XCSF.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    store(struct XCSF *xcsf)
    {
        xcsf_store_pset(xcsf);
        if (verbose) {
            std::ostringstream status;
            status << "checkpoint: ";
            status << std::fixed << std::setprecision(5) << best_error;
            status << " error at " << best_trial << " trials";
            py::print(status.str());
        }
    }

    /**
     * @brief Restores the checkpointed XCSF.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    retrieve(struct XCSF *xcsf)
    {
        xcsf_retrieve_pset(xcsf);
        if (verbose) {
            std::ostringstream status;
            status << "restoring system from trial " << best_trial;
            status << " with error=";
            status << std::fixed << std::setprecision(5) << best_error;
            py::print(status.str());
        }
    }

    /**
     * @brief Checks whether early stopping criteria has been met.
     * @param [in] xcsf The XCSF data structure.
     * @param [in] metrics Dictionary of performance metrics.
     * @return whether early stopping criteria has been met.
     */
    bool
    should_stop(struct XCSF *xcsf, py::dict metrics)
    {
        py::list data = metrics[monitor];
        py::list trials = metrics["trials"];
        const double current_error = py::cast<double>(data[data.size() - 1]);
        const int current_trial = py::cast<int>(trials[trials.size() - 1]);
        if (current_trial < start_from) {
            return false;
        }
        if (current_error < best_error - min_delta) {
            best_error = current_error;
            best_trial = current_trial;
            if (restore) {
                store(xcsf);
            }
        }
        if (current_trial - patience > best_trial) {
            if (restore) {
                retrieve(xcsf);
            }
            return true;
        }
        return false;
    }

  private:
    py::str monitor;
    int patience;
    bool restore;
    double min_delta;
    int start_from;
    bool verbose;

    double best_error = std::numeric_limits<double>::max();
    int best_trial = 0;
};
