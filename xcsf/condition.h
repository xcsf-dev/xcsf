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

void condition_set(XCSF *xcsf, CL *c);

struct CondVtbl {
    _Bool (*cond_impl_crossover)(XCSF *xcsf, CL *c1, CL *c2);
    _Bool (*cond_impl_general)(XCSF *xcsf, CL *c1, CL *c2);
    _Bool (*cond_impl_match)(XCSF *xcsf, CL *c, double *x);
    _Bool (*cond_impl_mutate)(XCSF *xcsf, CL *c);
    void (*cond_impl_copy)(XCSF *xcsf, CL *to, CL *from);
    void (*cond_impl_cover)(XCSF *xcsf, CL *c, double *x);
    void (*cond_impl_free)(XCSF *xcsf, CL *c);
    void (*cond_impl_init)(XCSF *xcsf, CL *c);
    void (*cond_impl_print)(XCSF *xcsf, CL *c);
    void (*cond_impl_update)(XCSF *xcsf, CL *c, double *x, double *y);
    int (*cond_impl_size)(XCSF *xcsf, CL *c);
    size_t (*cond_impl_save)(XCSF *xcsf, CL *c, FILE *fp);
    size_t (*cond_impl_load)(XCSF *xcsf, CL *c, FILE *fp);
};

static inline size_t cond_save(XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->cond_vptr->cond_impl_save)(xcsf, c, fp);
}

static inline size_t cond_load(XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->cond_vptr->cond_impl_load)(xcsf, c, fp);
}

static inline int cond_size(XCSF *xcsf, CL *c) {
    return (*c->cond_vptr->cond_impl_size)(xcsf, c);
}
 
static inline void cond_update(XCSF *xcsf, CL *c, double *x, double *y) {
    (*c->cond_vptr->cond_impl_update)(xcsf, c, x, y);
}

static inline _Bool cond_crossover(XCSF *xcsf, CL *c1, CL *c2) {
    return (*c1->cond_vptr->cond_impl_crossover)(xcsf, c1, c2);
}

static inline _Bool cond_general(XCSF *xcsf, CL *c1, CL *c2) {
    return (*c1->cond_vptr->cond_impl_general)(xcsf, c1, c2);
}

static inline _Bool cond_match(XCSF *xcsf, CL *c, double *x) {
    return (*c->cond_vptr->cond_impl_match)(xcsf, c, x);
}

static inline _Bool cond_mutate(XCSF *xcsf, CL *c) {
    return (*c->cond_vptr->cond_impl_mutate)(xcsf, c);
}

static inline void cond_copy(XCSF *xcsf, CL *to, CL *from) {
    (*from->cond_vptr->cond_impl_copy)(xcsf, to, from);
}

static inline void cond_cover(XCSF *xcsf, CL *c, double *x) {
    (*c->cond_vptr->cond_impl_cover)(xcsf, c, x);
}

static inline void cond_free(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_free)(xcsf, c);
}

static inline void cond_init(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_init)(xcsf, c);
}

static inline void cond_print(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_print)(xcsf, c);
}
