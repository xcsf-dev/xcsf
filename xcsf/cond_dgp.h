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

_Bool cond_dgp_crossover(CL *c1, CL *c2);
_Bool cond_dgp_general(CL *c1, CL *c2);
_Bool cond_dgp_match(CL *c, double *x);
_Bool cond_dgp_match_state(CL *c);
_Bool cond_dgp_mutate(CL *c);
_Bool cond_dgp_subsumes(CL *c1, CL *c2);
void cond_dgp_copy(CL *to, CL *from);
void cond_dgp_cover(CL *c, double *x);
void cond_dgp_free(CL *c);
void cond_dgp_init(CL *c);
void cond_dgp_print(CL *c);
void cond_dgp_rand(CL *c);
double cond_dgp_mu(CL *c, int m);

static struct CondVtbl const cond_dgp_vtbl = {
	&cond_dgp_crossover,
	&cond_dgp_general,
	&cond_dgp_match,
	&cond_dgp_match_state,
	&cond_dgp_mutate,
	&cond_dgp_subsumes,
	&cond_dgp_mu,
	&cond_dgp_copy,
	&cond_dgp_cover,
	&cond_dgp_free,
	&cond_dgp_init,
	&cond_dgp_print,
	&cond_dgp_rand
};      
