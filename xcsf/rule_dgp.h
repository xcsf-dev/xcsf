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
 * @file rule_dgp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Dynamical GP graph rule (condition + action) functions.
 */

#pragma once

#include "action.h"
#include "condition.h"
#include "dgp.h"
#include "xcsf.h"

/**
 * @brief Dynamical GP graph rule data structure.
 */
struct RULE_DGP {
    struct GRAPH dgp; //!< DGP graph
    int n_outputs; //!< Number of action nodes (binarised)
};

_Bool
rule_dgp_cond_crossover(const struct XCSF *xcsf, const struct CL *c1,
                        const struct CL *c2);

_Bool
rule_dgp_cond_general(const struct XCSF *xcsf, const struct CL *c1,
                      const struct CL *c2);

_Bool
rule_dgp_cond_match(const struct XCSF *xcsf, const struct CL *c,
                    const double *x);

_Bool
rule_dgp_cond_mutate(const struct XCSF *xcsf, const struct CL *c);

void
rule_dgp_cond_copy(const struct XCSF *xcsf, struct CL *dest,
                   const struct CL *src);

void
rule_dgp_cond_cover(const struct XCSF *xcsf, const struct CL *c,
                    const double *x);

void
rule_dgp_cond_free(const struct XCSF *xcsf, const struct CL *c);

void
rule_dgp_cond_init(const struct XCSF *xcsf, struct CL *c);

void
rule_dgp_cond_print(const struct XCSF *xcsf, const struct CL *c);

void
rule_dgp_cond_update(const struct XCSF *xcsf, const struct CL *c,
                     const double *x, const double *y);

double
rule_dgp_cond_size(const struct XCSF *xcsf, const struct CL *c);

size_t
rule_dgp_cond_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp);

size_t
rule_dgp_cond_load(const struct XCSF *xcsf, struct CL *c, FILE *fp);

/**
 * @brief Dynamical GP rule condition implemented functions.
 */
static struct CondVtbl const rule_dgp_cond_vtbl = {
    &rule_dgp_cond_crossover, &rule_dgp_cond_general, &rule_dgp_cond_match,
    &rule_dgp_cond_mutate,    &rule_dgp_cond_copy,    &rule_dgp_cond_cover,
    &rule_dgp_cond_free,      &rule_dgp_cond_init,    &rule_dgp_cond_print,
    &rule_dgp_cond_update,    &rule_dgp_cond_size,    &rule_dgp_cond_save,
    &rule_dgp_cond_load
};

_Bool
rule_dgp_act_crossover(const struct XCSF *xcsf, const struct CL *c1,
                       const struct CL *c2);

_Bool
rule_dgp_act_general(const struct XCSF *xcsf, const struct CL *c1,
                     const struct CL *c2);

_Bool
rule_dgp_act_mutate(const struct XCSF *xcsf, const struct CL *c);

int
rule_dgp_act_compute(const struct XCSF *xcsf, const struct CL *c,
                     const double *x);

void
rule_dgp_act_copy(const struct XCSF *xcsf, struct CL *dest,
                  const struct CL *src);

void
rule_dgp_act_cover(const struct XCSF *xcsf, const struct CL *c, const double *x,
                   const int action);

void
rule_dgp_act_free(const struct XCSF *xcsf, const struct CL *c);

void
rule_dgp_act_init(const struct XCSF *xcsf, struct CL *c);

void
rule_dgp_act_print(const struct XCSF *xcsf, const struct CL *c);

void
rule_dgp_act_update(const struct XCSF *xcsf, const struct CL *c,
                    const double *x, const double *y);

size_t
rule_dgp_act_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp);

size_t
rule_dgp_act_load(const struct XCSF *xcsf, struct CL *c, FILE *fp);

/**
 * @brief Dynamical GP rule action implemented functions.
 */
static struct ActVtbl const rule_dgp_act_vtbl = {
    &rule_dgp_act_general, &rule_dgp_act_crossover, &rule_dgp_act_mutate,
    &rule_dgp_act_compute, &rule_dgp_act_copy,      &rule_dgp_act_cover,
    &rule_dgp_act_free,    &rule_dgp_act_init,      &rule_dgp_act_print,
    &rule_dgp_act_update,  &rule_dgp_act_save,      &rule_dgp_act_load
};
