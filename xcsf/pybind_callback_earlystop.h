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
 * @file pybind_callback_earlystop.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Early stopping callback for Python library.
 */

#pragma once

#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
#include "xcsf.h"
}

#include "pybind_callback.h"
#include "pybind_utils.h"

class EarlyStoppingCallback : public Callback
{
  public:
    /**
     * @brief Constructs a new early stopping callback.
     * @param [in] monitor Name of the metric to monitor: {"train", "val"}.
     * @param [in] patience Trials with no improvement after which training will
     * be stopped.
     * @param [in] restore Whether to restore the best population.
     * @param [in] min_delta Minimum change to qualify as an improvement.
     * @param [in] start_from Trials to wait before starting to monitor
     * improvement.
     * @param [in] verbose Whether to display messages when an action is taken.
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
            err << "patience cannot be negative" << std::endl;
            throw std::invalid_argument(err.str());
        }
        if (min_delta < 0) {
            err << "min_delta cannot be negative" << std::endl;
            throw std::invalid_argument(err.str());
        }
    }

    /**
     * @brief Stores best XCSF population in memory.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    store(struct XCSF *xcsf)
    {
        do_restore = true;
        xcsf_store_pset(xcsf);
        if (verbose) {
            std::ostringstream status;
            status << get_timestamp() << " EarlyStoppingCallback: ";
            status << std::fixed << std::setprecision(5) << best_error;
            status << " best error at " << best_trial << " trials";
            py::print(status.str());
        }
    }

    /**
     * @brief Retrieves best XCSF population in memory.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    retrieve(struct XCSF *xcsf)
    {
        do_restore = false;
        xcsf_retrieve_pset(xcsf);
        if (verbose) {
            std::ostringstream status;
            status << get_timestamp() << " EarlyStoppingCallback: ";
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
    run(struct XCSF *xcsf, py::dict metrics) override
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
            if (verbose) {
                std::ostringstream status;
                status << get_timestamp() << " EarlyStoppingCallback: stopping";
                py::print(status.str());
            }
            if (restore) {
                retrieve(xcsf);
            }
            return true;
        }
        return false;
    }

    /**
     * @brief Executes any tasks at the end of fitting.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    finish(struct XCSF *xcsf) override
    {
        if (restore && do_restore) {
            retrieve(xcsf);
        }
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
    bool do_restore = false;
};
