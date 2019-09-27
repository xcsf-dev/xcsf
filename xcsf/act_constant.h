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

_Bool act_constant_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool act_constant_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool act_constant_mutate(XCSF *xcsf, CL *c);
double *act_constant_compute(XCSF *xcsf, CL *c, double *x);
void act_constant_copy(XCSF *xcsf, CL *to, CL *from);
void act_constant_free(XCSF *xcsf, CL *c);
void act_constant_init(XCSF *xcsf, CL *c);
void act_constant_print(XCSF *xcsf, CL *c);
void act_constant_rand(XCSF *xcsf, CL *c);
void act_constant_update(XCSF *xcsf, CL *c, double *x, double *y);
void act_constant_save(XCSF *xcsf, CL *c, FILE *fp);
void act_constant_load(XCSF *xcsf, CL *c, FILE *fp);

static struct ActVtbl const act_constant_vtbl = {
    &act_constant_general,
    &act_constant_crossover,
    &act_constant_mutate,
    &act_constant_compute,
    &act_constant_copy,
    &act_constant_free,
    &act_constant_init,
    &act_constant_rand,
    &act_constant_print,
    &act_constant_update,
    &act_constant_save,
    &act_constant_load
};     
