/*
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

/**
 * @file cl.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Functions operating on classifiers.
 */

#pragma once

#include "xcsf.h"

_Bool
cl_crossover(const struct XCSF *xcsf, const struct CL *c1, const struct CL *c2);

_Bool
cl_general(const struct XCSF *xcsf, const struct CL *c1, const struct CL *c2);

_Bool
cl_m(const struct XCSF *xcsf, const struct CL *c);

_Bool
cl_match(const struct XCSF *xcsf, struct CL *c, const double *x);

_Bool
cl_mutate(const struct XCSF *xcsf, const struct CL *c);

_Bool
cl_subsumer(const struct XCSF *xcsf, const struct CL *c);

const double *
cl_predict(const struct XCSF *xcsf, const struct CL *c, const double *x);

double
cl_acc(const struct XCSF *xcsf, const struct CL *c);

double
cl_del_vote(const struct XCSF *xcsf, const struct CL *c, const double avg_fit);

double
cl_mfrac(const struct XCSF *xcsf, const struct CL *c);

int
cl_action(const struct XCSF *xcsf, struct CL *c, const double *x);

double
cl_cond_size(const struct XCSF *xcsf, const struct CL *c);

double
cl_pred_size(const struct XCSF *xcsf, const struct CL *c);

size_t
cl_load(const struct XCSF *xcsf, struct CL *c, FILE *fp);

size_t
cl_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp);

void
cl_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src);

void
cl_cover(const struct XCSF *xcsf, struct CL *c, const double *x,
         const int action);

void
cl_free(const struct XCSF *xcsf, struct CL *c);

void
cl_init(const struct XCSF *xcsf, struct CL *c, const double size,
        const int time);

void
cl_init_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src);

void
cl_print(const struct XCSF *xcsf, const struct CL *c, const _Bool print_cond,
         const _Bool print_act, const _Bool print_pred);

void
cl_rand(const struct XCSF *xcsf, struct CL *c);

void
cl_update(const struct XCSF *xcsf, struct CL *c, const double *x,
          const double *y, const int set_num, const _Bool cur);

void
cl_update_fit(const struct XCSF *xcsf, struct CL *c, const double acc_sum,
              const double acc);
