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

_Bool rule_dgp_cond_crossover(CL *c1, CL *c2);
_Bool rule_dgp_cond_general(CL *c1, CL *c2);
_Bool rule_dgp_cond_match(CL *c, double *x);
_Bool rule_dgp_cond_match_state(CL *c);
_Bool rule_dgp_cond_mutate(CL *c);
_Bool rule_dgp_cond_subsumes(CL *c1, CL *c2);
void rule_dgp_cond_copy(CL *to, CL *from);
void rule_dgp_cond_cover(CL *c, double *x);
void rule_dgp_cond_free(CL *c);
void rule_dgp_cond_init(CL *c);
void rule_dgp_cond_print(CL *c);
void rule_dgp_cond_rand(CL *c);
double rule_dgp_cond_mu(CL *c, int m);

static struct CondVtbl const rule_dgp_cond_vtbl = {
	&rule_dgp_cond_crossover,
	&rule_dgp_cond_general,
	&rule_dgp_cond_match,
	&rule_dgp_cond_match_state,
	&rule_dgp_cond_mutate,
	&rule_dgp_cond_subsumes,
	&rule_dgp_cond_mu,
	&rule_dgp_cond_copy,
	&rule_dgp_cond_cover,
	&rule_dgp_cond_free,
	&rule_dgp_cond_init,
	&rule_dgp_cond_print,
	&rule_dgp_cond_rand
};      

double rule_dgp_pred_pre(CL *c, int p);
double *rule_dgp_pred_compute(CL *c, double *x);
void rule_dgp_pred_copy(CL *to,  CL *from);
void rule_dgp_pred_free(CL *c);
void rule_dgp_pred_init(CL *c);
void rule_dgp_pred_print(CL *c);
void rule_dgp_pred_update(CL *c, double *y, double *x);

static struct PredVtbl const rule_dgp_pred_vtbl = {
	&rule_dgp_pred_compute,
	&rule_dgp_pred_pre,
	&rule_dgp_pred_copy,
	&rule_dgp_pred_free,
	&rule_dgp_pred_init,
	&rule_dgp_pred_print,
	&rule_dgp_pred_update
};
