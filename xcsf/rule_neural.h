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
 * @file rule_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Neural network rule (condition + action) functions.
 */ 

#pragma once

_Bool rule_neural_cond_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_cond_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_cond_match(XCSF *xcsf, CL *c, const double *x);
_Bool rule_neural_cond_mutate(XCSF *xcsf, CL *c);
void rule_neural_cond_copy(XCSF *xcsf, CL *to, CL *from);
void rule_neural_cond_cover(XCSF *xcsf, CL *c, const double *x);
void rule_neural_cond_free(XCSF *xcsf, CL *c);
void rule_neural_cond_init(XCSF *xcsf, CL *c);
void rule_neural_cond_print(XCSF *xcsf, CL *c);
void rule_neural_cond_update(XCSF *xcsf, CL *c, const double *x, const double *y);
int rule_neural_cond_size(XCSF *xcsf, CL *c);
size_t rule_neural_cond_save(XCSF *xcsf, CL *c, FILE *fp);
size_t rule_neural_cond_load(XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Neural network rule condition implemented functions.
 */
static struct CondVtbl const rule_neural_cond_vtbl = {
    &rule_neural_cond_crossover,
    &rule_neural_cond_general,
    &rule_neural_cond_match,
    &rule_neural_cond_mutate,
    &rule_neural_cond_copy,
    &rule_neural_cond_cover,
    &rule_neural_cond_free,
    &rule_neural_cond_init,
    &rule_neural_cond_print,
    &rule_neural_cond_update,
    &rule_neural_cond_size,
    &rule_neural_cond_save,
    &rule_neural_cond_load
};      

_Bool rule_neural_act_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_act_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_act_mutate(XCSF *xcsf, CL *c);
int rule_neural_act_compute(XCSF *xcsf, CL *c, const double *x);
void rule_neural_act_copy(XCSF *xcsf, CL *to, CL *from);
void rule_neural_act_cover(XCSF *xcsf, CL *c, const double *x, int action);
void rule_neural_act_free(XCSF *xcsf, CL *c);
void rule_neural_act_init(XCSF *xcsf, CL *c);
void rule_neural_act_print(XCSF *xcsf, CL *c);
void rule_neural_act_rand(XCSF *xcsf, CL *c);
void rule_neural_act_update(XCSF *xcsf, CL *c, const double *x, const double *y);
size_t rule_neural_act_save(XCSF *xcsf, CL *c, FILE *fp);
size_t rule_neural_act_load(XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Neural network rule action implemented functions.
 */
static struct ActVtbl const rule_neural_act_vtbl = {
    &rule_neural_act_general,
    &rule_neural_act_crossover,
    &rule_neural_act_mutate,
    &rule_neural_act_compute,
    &rule_neural_act_copy,
    &rule_neural_act_cover,
    &rule_neural_act_free,
    &rule_neural_act_init,
    &rule_neural_act_rand,
    &rule_neural_act_print,
    &rule_neural_act_update,
    &rule_neural_act_save,
    &rule_neural_act_load
};
