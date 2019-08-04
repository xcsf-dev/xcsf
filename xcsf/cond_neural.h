/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
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

_Bool cond_neural_crossover(CL *c1, CL *c2);
_Bool cond_neural_general(CL *c1, CL *c2);
_Bool cond_neural_match(CL *c, double *x);
_Bool cond_neural_match_state(CL *c);
_Bool cond_neural_mutate(CL *c);
_Bool cond_neural_subsumes(CL *c1, CL *c2);
void cond_neural_copy(CL *to, CL *from);
void cond_neural_cover(CL *c, double *x);
void cond_neural_free(CL *c);
void cond_neural_init(CL *c);
void cond_neural_print(CL *c);
void cond_neural_rand(CL *c);
double cond_neural_mu(CL *c, int m);

static struct CondVtbl const cond_neural_vtbl = {
	&cond_neural_crossover,
	&cond_neural_general,
	&cond_neural_match,
	&cond_neural_match_state,
	&cond_neural_mutate,
	&cond_neural_subsumes,
	&cond_neural_mu,
	&cond_neural_copy,
	&cond_neural_cover,
	&cond_neural_free,
	&cond_neural_init,
	&cond_neural_print,
	&cond_neural_rand
};      
