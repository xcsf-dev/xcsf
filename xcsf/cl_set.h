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
 * @file cl_set.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Functions operating on sets of classifiers.
 */ 

#pragma once
 
double set_mean_inputs_matched(XCSF *xcsf, SET *set);
double set_mean_cond_size(XCSF *xcsf, SET *set);
double set_mean_eta(XCSF *xcsf, SET *set, int layer);
double set_mean_mut(XCSF *xcsf, SET *set, int m);
double set_mean_pred_size(XCSF *xcsf, SET *set);
double set_mean_time(XCSF *xcsf, SET *set);
double set_total_fit(XCSF *xcsf, SET *set);
size_t pop_load(XCSF *xcsf, FILE *fp);
size_t pop_save(XCSF *xcsf, FILE *fp);
void pop_enforce_limit(XCSF *xcsf, SET *kset);
void pop_init(XCSF *xcsf);
void set_action(XCSF *xcsf, SET *mset, SET *aset, int action);
void set_add(XCSF *xcsf, SET *set, CL *c);
void set_free(XCSF *xcsf, SET *set);
void set_init(XCSF *xcsf, SET *set);
void set_kill(XCSF *xcsf, SET *set);
void set_match(XCSF *xcsf, SET *mset, SET *kset, double *x);
void set_pred(XCSF *xcsf, SET *set, double *x, double *p);
void set_print(XCSF *xcsf, SET *set, _Bool printc, _Bool printa, _Bool printp);
void set_times(XCSF *xcsf, SET *set);
void set_update(XCSF *xcsf, SET *set, SET *kset, double *x, double *y, _Bool current);
void set_validate(XCSF *xcsf, SET *set);
