 /*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
 *
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

_Bool rule_neural_cond_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_cond_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_cond_match(XCSF *xcsf, CL *c, double *x);
_Bool rule_neural_cond_mutate(XCSF *xcsf, CL *c);
void rule_neural_cond_copy(XCSF *xcsf, CL *to, CL *from);
void rule_neural_cond_cover(XCSF *xcsf, CL *c, double *x);
void rule_neural_cond_free(XCSF *xcsf, CL *c);
void rule_neural_cond_init(XCSF *xcsf, CL *c);
void rule_neural_cond_print(XCSF *xcsf, CL *c);
void rule_neural_cond_rand(XCSF *xcsf, CL *c);

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
    &rule_neural_cond_rand
};      

double *rule_neural_pred_compute(XCSF *xcsf, CL *c, double *x);
_Bool rule_neural_pred_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool rule_neural_pred_mutate(XCSF *xcsf, CL *c);
void rule_neural_pred_copy(XCSF *xcsf, CL *to,  CL *from);
void rule_neural_pred_free(XCSF *xcsf, CL *c);
void rule_neural_pred_init(XCSF *xcsf, CL *c);
void rule_neural_pred_print(XCSF *xcsf, CL *c);
void rule_neural_pred_update(XCSF *xcsf, CL *c, double *x, double *y);

static struct PredVtbl const rule_neural_pred_vtbl = {
    &rule_neural_pred_crossover,
    &rule_neural_pred_mutate,
    &rule_neural_pred_compute,
    &rule_neural_pred_copy,
    &rule_neural_pred_free,
    &rule_neural_pred_init,
    &rule_neural_pred_print,
    &rule_neural_pred_update
};
