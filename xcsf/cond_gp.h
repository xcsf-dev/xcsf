/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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
 *
 */

_Bool cond_gp_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_gp_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_gp_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_gp_match_state(XCSF *xcsf, CL *c);
_Bool cond_gp_mutate(XCSF *xcsf, CL *c);
void cond_gp_copy(XCSF *xcsf, CL *to, CL *from);
void cond_gp_cover(XCSF *xcsf, CL *c, double *x);
void cond_gp_free(XCSF *xcsf, CL *c);
void cond_gp_init(XCSF *xcsf, CL *c);
void cond_gp_print(XCSF *xcsf, CL *c);
void cond_gp_rand(XCSF *xcsf, CL *c);
double cond_gp_mu(XCSF *xcsf, CL *c, int m);

static struct CondVtbl const cond_gp_vtbl = {
	&cond_gp_crossover,
	&cond_gp_general,
	&cond_gp_match,
	&cond_gp_match_state,
	&cond_gp_mutate,
	&cond_gp_mu,
	&cond_gp_copy,
	&cond_gp_cover,
	&cond_gp_free,
	&cond_gp_init,
	&cond_gp_print,
	&cond_gp_rand
};      
