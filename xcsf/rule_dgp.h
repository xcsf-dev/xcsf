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

/**
 * @brief Dynamical GP graph rule data structure.
 */ 
typedef struct RULE_DGP{
    GRAPH dgp; //!< DGP graph
    int n_outputs; //!< Number of action nodes (binarised)
} RULE_DGP;

_Bool rule_dgp_cond_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_dgp_cond_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_dgp_cond_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool rule_dgp_cond_mutate(const XCSF *xcsf, const CL *c);
void rule_dgp_cond_copy(const XCSF *xcsf, CL *dest, const CL *src);
void rule_dgp_cond_cover(const XCSF *xcsf, const CL *c, const double *x);
void rule_dgp_cond_free(const XCSF *xcsf, const CL *c);
void rule_dgp_cond_init(const XCSF *xcsf, CL *c);
void rule_dgp_cond_print(const XCSF *xcsf, const CL *c);
void rule_dgp_cond_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int rule_dgp_cond_size(const XCSF *xcsf, const CL *c);
size_t rule_dgp_cond_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t rule_dgp_cond_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Dynamical GP rule condition implemented functions.
 */
static struct CondVtbl const rule_dgp_cond_vtbl = {
    &rule_dgp_cond_crossover,
    &rule_dgp_cond_general,
    &rule_dgp_cond_match,
    &rule_dgp_cond_mutate,
    &rule_dgp_cond_copy,
    &rule_dgp_cond_cover,
    &rule_dgp_cond_free,
    &rule_dgp_cond_init,
    &rule_dgp_cond_print,
    &rule_dgp_cond_update,
    &rule_dgp_cond_size,
    &rule_dgp_cond_save,
    &rule_dgp_cond_load
};      

_Bool rule_dgp_act_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_dgp_act_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_dgp_act_mutate(const XCSF *xcsf, const CL *c);
int rule_dgp_act_compute(const XCSF *xcsf, const CL *c, const double *x);
void rule_dgp_act_copy(const XCSF *xcsf, CL *dest, const CL *src);
void rule_dgp_act_cover(const XCSF *xcsf, const CL *c, const double *x, int action);
void rule_dgp_act_free(const XCSF *xcsf, const CL *c);
void rule_dgp_act_init(const XCSF *xcsf, CL *c);
void rule_dgp_act_print(const XCSF *xcsf, const CL *c);
void rule_dgp_act_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
size_t rule_dgp_act_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t rule_dgp_act_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Dynamical GP rule action implemented functions.
 */
static struct ActVtbl const rule_dgp_act_vtbl = {
    &rule_dgp_act_general,
    &rule_dgp_act_crossover,
    &rule_dgp_act_mutate,
    &rule_dgp_act_compute,
    &rule_dgp_act_copy,
    &rule_dgp_act_cover,
    &rule_dgp_act_free,
    &rule_dgp_act_init,
    &rule_dgp_act_print,
    &rule_dgp_act_update,
    &rule_dgp_act_save,
    &rule_dgp_act_load
};
