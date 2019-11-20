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
#pragma once

_Bool cond_dummy_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dummy_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dummy_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_dummy_mutate(XCSF *xcsf, CL *c);
void cond_dummy_copy(XCSF *xcsf, CL *to, CL *from);
void cond_dummy_cover(XCSF *xcsf, CL *c, double *x);
void cond_dummy_free(XCSF *xcsf, CL *c);
void cond_dummy_init(XCSF *xcsf, CL *c);
void cond_dummy_print(XCSF *xcsf, CL *c);
void cond_dummy_update(XCSF *xcsf, CL *c, double *x, double *y);
int cond_dummy_size(XCSF *xcsf, CL *c);
size_t cond_dummy_save(XCSF *xcsf, CL *c, FILE *fp);
size_t cond_dummy_load(XCSF *xcsf, CL *c, FILE *fp);

static struct CondVtbl const cond_dummy_vtbl = {
    &cond_dummy_crossover,
    &cond_dummy_general,
    &cond_dummy_match,
    &cond_dummy_mutate,
    &cond_dummy_copy,
    &cond_dummy_cover,
    &cond_dummy_free,
    &cond_dummy_init,
    &cond_dummy_print,
    &cond_dummy_update,
    &cond_dummy_size,
    &cond_dummy_save,
    &cond_dummy_load
};
