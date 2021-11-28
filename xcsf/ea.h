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
 * @file ea.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Evolutionary algorithm functions.
 */

#pragma once

#include "xcsf.h"

#define EA_SELECT_ROULETTE (0) //!< Roulette wheel parental selection
#define EA_SELECT_TOURNAMENT (1) //!< Tournament parental selection

#define EA_STRING_ROULETTE ("roulette\0") //!< Roulette
#define EA_STRING_TOURNAMENT ("tournament\0") //!< Tournament

/**
 * @brief Parameters for operating the evolutionary algorithm.
 */
struct ArgsEA {
    bool subsumption; //!< Whether to try and subsume offspring classifiers
    double select_size; //!< Fraction of set size for tournaments
    double err_reduc; //!< Amount to reduce an offspring's error
    double fit_reduc; //!< Amount to reduce an offspring's fitness
    double p_crossover; //!< Probability of applying crossover
    double theta; //!< Average match set time between EA invocations
    int lambda; //!< Number of offspring to create each EA invocation
    int select_type; //!< Roulette or tournament for EA parental selection
    bool pred_reset; //!< Whether to reset or copy offspring predictions
};

void
ea(struct XCSF *xcsf, const struct Set *set);

void
ea_param_defaults(struct XCSF *xcsf);

const char *
ea_param_json_export(const struct XCSF *xcsf);

size_t
ea_param_save(const struct XCSF *xcsf, FILE *fp);

size_t
ea_param_load(struct XCSF *xcsf, FILE *fp);

const char *
ea_type_as_string(const int type);

int
ea_type_as_int(const char *type);

/* parameter setters */

void
ea_param_set_select_size(struct XCSF *xcsf, const double a);

void
ea_param_set_theta(struct XCSF *xcsf, const double a);

void
ea_param_set_p_crossover(struct XCSF *xcsf, const double a);

void
ea_param_set_lambda(struct XCSF *xcsf, const int a);

void
ea_param_set_err_reduc(struct XCSF *xcsf, const double a);

void
ea_param_set_fit_reduc(struct XCSF *xcsf, const double a);

void
ea_param_set_subsumption(struct XCSF *xcsf, const bool a);

void
ea_param_set_pred_reset(struct XCSF *xcsf, const bool a);

void
ea_param_set_select_type(struct XCSF *xcsf, const int a);

void
ea_param_set_type_string(struct XCSF *xcsf, const char *a);
