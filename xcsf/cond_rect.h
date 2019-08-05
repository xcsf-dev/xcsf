 /*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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

_Bool cond_rect_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_rect_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_rect_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_rect_match_state(XCSF *xcsf, CL *c);
_Bool cond_rect_mutate(XCSF *xcsf, CL *c);
_Bool cond_rect_subsumes(XCSF *xcsf, CL *c1, CL *c2);
void cond_rect_copy(XCSF *xcsf, CL *to, CL *from);
void cond_rect_cover(XCSF *xcsf, CL *c, double *x);
void cond_rect_free(XCSF *xcsf, CL *c);
void cond_rect_init(XCSF *xcsf, CL *c);
void cond_rect_print(XCSF *xcsf, CL *c);
void cond_rect_rand(XCSF *xcsf, CL *c);
double cond_rect_mu(XCSF *xcsf, CL *c, int m);

static struct CondVtbl const cond_rect_vtbl = {
	&cond_rect_crossover,
	&cond_rect_general,
	&cond_rect_match,
	&cond_rect_match_state,
	&cond_rect_mutate,
	&cond_rect_subsumes,
	&cond_rect_mu,
	&cond_rect_copy,
	&cond_rect_cover,
	&cond_rect_free,
	&cond_rect_init,
	&cond_rect_print,
	&cond_rect_rand
};      
