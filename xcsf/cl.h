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
 * @date 2015--2021.
 * @brief Functions operating on classifiers.
 */

#pragma once

#include "xcsf.h"

bool
cl_crossover(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2);

bool
cl_general(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2);

bool
cl_m(const struct XCSF *xcsf, const struct Cl *c);

bool
cl_match(const struct XCSF *xcsf, struct Cl *c, const double *x);

bool
cl_mutate(const struct XCSF *xcsf, const struct Cl *c);

bool
cl_subsumer(const struct XCSF *xcsf, const struct Cl *c);

const double *
cl_predict(const struct XCSF *xcsf, const struct Cl *c, const double *x);

double
cl_acc(const struct XCSF *xcsf, const struct Cl *c);

double
cl_del_vote(const struct XCSF *xcsf, const struct Cl *c, const double avg_fit);

double
cl_mfrac(const struct XCSF *xcsf, const struct Cl *c);

int
cl_action(const struct XCSF *xcsf, struct Cl *c, const double *x);

double
cl_cond_size(const struct XCSF *xcsf, const struct Cl *c);

double
cl_pred_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
cl_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

size_t
cl_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

void
cl_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
cl_cover(const struct XCSF *xcsf, struct Cl *c, const double *x,
         const int action);

void
cl_free(const struct XCSF *xcsf, struct Cl *c);

void
cl_init(const struct XCSF *xcsf, struct Cl *c, const double size,
        const int time);

void
cl_init_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
cl_print(const struct XCSF *xcsf, const struct Cl *c, const bool print_cond,
         const bool print_act, const bool print_pred);

void
cl_rand(const struct XCSF *xcsf, struct Cl *c);

void
cl_update(const struct XCSF *xcsf, struct Cl *c, const double *x,
          const double *y, const int set_num, const bool cur);

void
cl_update_fit(const struct XCSF *xcsf, struct Cl *c, const double acc_sum,
              const double acc);

const char *
cl_json_export(const struct XCSF *xcsf, const struct Cl *c,
               const bool return_cond, const bool return_act,
               const bool return_pred);
