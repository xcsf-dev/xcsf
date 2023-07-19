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
 * @file pybind_callback_checkpoint.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Checkpoint callback for Python library.
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

class CheckpointCallback : public Callback
{
  public:
    /**
     * @brief Constructs a new checkpoint callback.
     * @param [in] monitor Name of the metric to monitor: {"train", "val"}.
     * @param [in] filename Name of the file to save XCSF.
     * @param [in] save_best_only Whether to only save the best population.
     * @param [in] save_freq Trial frequency to (possibly) make checkpoints.
     * @param [in] verbose Whether to display messages when an action is taken.
     */
    CheckpointCallback(py::str monitor, std::string filename,
                       bool save_best_only, int save_freq, bool verbose) :
        monitor(monitor),
        filename(filename),
        save_best_only(save_best_only),
        save_freq(save_freq),
        verbose(verbose)
    {
        std::ostringstream err;
        std::string str = monitor.cast<std::string>();
        if (str != "train" && str != "val") {
            err << "invalid metric to monitor: " << str << std::endl;
            throw std::invalid_argument(err.str());
        }
        if (save_freq < 0) {
            err << "save_freq cannot be negative" << std::endl;
            throw std::invalid_argument(err.str());
        }
    }

    /**
     * @brief Saves the state of XCSF.
     * @param [in] xcsf The XCSF data structure.
     */
    void
    save(struct XCSF *xcsf)
    {
        xcsf_save(xcsf, filename.c_str());
        std::ostringstream status;
        status << get_timestamp() << " CheckpointCallback: ";
        status << "saved " << filename;
        py::print(status.str());
    }

    /**
     * @brief Performs callback operations.
     * @param [in] xcsf The XCSF data structure.
     * @param [in] metrics Dictionary of performance metrics.
     * @return Whether to terminate training.
     */
    bool
    run(struct XCSF *xcsf, py::dict metrics) override
    {
        py::list data = metrics[monitor];
        py::list trials = metrics["trials"];
        const double current_error = py::cast<double>(data[data.size() - 1]);
        const int current_trial = py::cast<int>(trials[trials.size() - 1]);
        if (current_trial >= save_trial + save_freq) {
            if (!save_best_only || (current_error < best_error)) {
                save_trial = current_trial;
                save(xcsf);
            }
            if (current_error < best_error) {
                best_error = current_error;
            }
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
        if (!save_best_only) {
            save(xcsf);
        }
    }

  private:
    py::str monitor;
    std::string filename;
    bool save_best_only;
    int save_freq;
    bool verbose;

    double best_error = std::numeric_limits<double>::max();
    int save_trial = 0;
};
