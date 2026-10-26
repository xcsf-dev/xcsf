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
 * @file pybind_utils.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023--2026.
 * @brief Utilities for Python library.
 */

#pragma once

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

extern "C" {
#include "xcsf.h"
}

/**
 * @brief Returns a formatted string for displaying time.
 * @return String representation of the current time.
 */
std::string
get_timestamp()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t current = system_clock::to_time_t(now);
    std::tm local = *std::localtime(&current);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &local);
    auto dur = now.time_since_epoch();
    auto ms = duration_cast<milliseconds>(dur) % 1000;
    std::ostringstream oss;
    oss << buffer << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

static void
check_status(struct XCSF *xcsf)
{
    if (xcsf->status == XCSF_OK) {
        return;
    }

    const std::string msg(xcsf->err_msg);
    const int code = xcsf->status;
    xcsf_clear_error(xcsf);

    switch (code) {
        case XCSF_ERROR:
            throw std::invalid_argument(msg);
        default:
            throw std::runtime_error(msg);
    }
}
