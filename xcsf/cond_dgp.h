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
#pragma once

_Bool cond_dgp_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dgp_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dgp_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_dgp_mutate(XCSF *xcsf, CL *c);
void cond_dgp_copy(XCSF *xcsf, CL *to, CL *from);
void cond_dgp_cover(XCSF *xcsf, CL *c, double *x);
void cond_dgp_free(XCSF *xcsf, CL *c);
void cond_dgp_init(XCSF *xcsf, CL *c);
void cond_dgp_print(XCSF *xcsf, CL *c);
void cond_dgp_update(XCSF *xcsf, CL *c, double *x, double *y);
int cond_dgp_size(XCSF *xcsf, CL *c);
size_t cond_dgp_save(XCSF *xcsf, CL *c, FILE *fp);
size_t cond_dgp_load(XCSF *xcsf, CL *c, FILE *fp);

static struct CondVtbl const cond_dgp_vtbl = {
    &cond_dgp_crossover,
    &cond_dgp_general,
    &cond_dgp_match,
    &cond_dgp_mutate,
    &cond_dgp_copy,
    &cond_dgp_cover,
    &cond_dgp_free,
    &cond_dgp_init,
    &cond_dgp_print,
    &cond_dgp_update,
    &cond_dgp_size,
    &cond_dgp_save,
    &cond_dgp_load
};
