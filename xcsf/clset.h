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
 * @file clset.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Functions operating on sets of classifiers.
 */ 

#pragma once
 
double clset_mean_cond_size(const XCSF *xcsf, const SET *set);
double clset_mean_eta(const XCSF *xcsf, const SET *set, int layer);
double clset_mean_inputs_matched(const XCSF *xcsf, const SET *set);
double clset_mean_mut(const XCSF *xcsf, const SET *set, int m);
double clset_mean_pred_size(const XCSF *xcsf, const SET *set);
double clset_mean_time(const XCSF *xcsf, const SET *set);
double clset_total_fit(const XCSF *xcsf, const SET *set);
size_t clset_pop_load(XCSF *xcsf, FILE *fp);
size_t clset_pop_save(const XCSF *xcsf, FILE *fp);
void clset_action(const XCSF *xcsf, const SET *mset, SET *aset, int action);
void clset_add(const XCSF *xcsf, SET *set, CL *c);
void clset_free(const XCSF *xcsf, SET *set);
void clset_init(const XCSF *xcsf, SET *set);
void clset_kill(XCSF *xcsf, SET *set);
void clset_match(XCSF *xcsf, SET *mset, SET *kset, const double *x);
void clset_pop_enforce_limit(XCSF *xcsf, SET *kset);
void clset_pop_init(XCSF *xcsf);
void clset_pred(const XCSF *xcsf, SET *set, const double *x, double *p);
void clset_print(const XCSF *xcsf, const SET *set, _Bool printc, _Bool printa, _Bool printp);
void clset_set_times(const XCSF *xcsf, SET *set);
void clset_update(XCSF *xcsf, SET *set, SET *kset, const double *x, const double *y, _Bool curr);
void clset_validate(const XCSF *xcsf, SET *set);
