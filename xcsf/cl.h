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

_Bool
cl_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);

_Bool
cl_general(const XCSF *xcsf, const CL *c1, const CL *c2);

_Bool
cl_m(const XCSF *xcsf, const CL *c);

_Bool
cl_match(const XCSF *xcsf, CL *c, const double *x);

_Bool
cl_mutate(const XCSF *xcsf, const CL *c);

_Bool
cl_subsumer(const XCSF *xcsf, const CL *c);

const double *
cl_predict(const XCSF *xcsf, const CL *c, const double *x);

double
cl_acc(const XCSF *xcsf, const CL *c);

double
cl_del_vote(const XCSF *xcsf, const CL *c, double avg_fit);

double
cl_mfrac(const XCSF *xcsf, const CL *c);

int
cl_action(const XCSF *xcsf, CL *c, const double *x);

int
cl_cond_size(const XCSF *xcsf, const CL *c);

int
cl_pred_size(const XCSF *xcsf, const CL *c);

size_t
cl_load(const XCSF *xcsf, CL *c, FILE *fp);

size_t
cl_save(const XCSF *xcsf, const CL *c, FILE *fp);

void
cl_copy(const XCSF *xcsf, CL *dest, const CL *src);

void
cl_cover(const XCSF *xcsf, CL *c, const double *x, int action);

void
cl_free(const XCSF *xcsf, CL *c);

void
cl_init(const XCSF *xcsf, CL *c, double size, int time);

void
cl_print(const XCSF *xcsf, const CL *c,
         _Bool printc, _Bool printa, _Bool printp);

void
cl_rand(const XCSF *xcsf, CL *c);

void
cl_update(const XCSF *xcsf, CL *c, const double *x, const double *y,
          int set_num, _Bool cur);

void
cl_update_fit(const XCSF *xcsf, CL *c, double acc_sum, double acc);
