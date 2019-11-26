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
 * @date 2015--2019.
 * @brief Functions operating on classifiers.
 */ 

#pragma once
 
_Bool cl_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cl_m(XCSF *xcsf, CL *c);
_Bool cl_match(XCSF *xcsf, CL *c, double *x);
_Bool cl_mutate(XCSF *xcsf, CL *c);
_Bool cl_subsumer(XCSF *xcsf, CL *c);
double *cl_predict(XCSF *xcsf, CL *c, double *x);
double cl_acc(XCSF *xcsf, CL *c);
double cl_del_vote(XCSF *xcsf, CL *c, double avg_fit);
double cl_mutation_rate(XCSF *xcsf, CL *c, int m);
void cl_copy(XCSF *xcsf, CL *to, CL *from);
void cl_cover(XCSF *xcsf, CL *c, double *x, int action);
void cl_free(XCSF *xcsf, CL *c);
void cl_init(XCSF *xcsf, CL *c, int size, int time);
void cl_print(XCSF *xcsf, CL *c, _Bool printc, _Bool printa, _Bool printp);
void cl_rand(XCSF *xcsf, CL *c);
void cl_update(XCSF *xcsf, CL *c, double *x, double *y, int set_num, _Bool current);
void cl_update_fit(XCSF *xcsf, CL *c, double acc_sum, double acc);
int cl_cond_size(XCSF *xcsf, CL *c);
int cl_pred_size(XCSF *xcsf, CL *c);
size_t cl_save(XCSF *xcsf, CL *c, FILE *fp);
size_t cl_load(XCSF *xcsf, CL *c, FILE *fp);
