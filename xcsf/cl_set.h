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
#pragma once

void set_init(XCSF *xcsf, SET *set);
double set_mean_time(XCSF *xcsf, SET *set);
double set_total_fit(XCSF *xcsf, SET *set);
double set_total_time(XCSF *xcsf, SET *set);
void pop_del(XCSF *xcsf, SET *kset);
void pop_enforce_limit(XCSF *xcsf, SET *kset);
void pop_init(XCSF *xcsf);
void set_add(XCSF *xcsf, SET *set, CL *c);
void set_free(XCSF *xcsf, SET *set);
void set_kill(XCSF *xcsf, SET *set);
void set_match(XCSF *xcsf, SET *mset, SET *kset, double *x);
void set_pred(XCSF *xcsf, SET *set, double *x, double *p);
void set_print(XCSF *xcsf, SET *set, _Bool printc, _Bool printa, _Bool printp);
void set_times(XCSF *xcsf, SET *set);
void set_update(XCSF *xcsf, SET *set, SET *kset, double *x, double *y);
void set_validate(XCSF *xcsf, SET *set);
double set_avg_mut(XCSF *xcsf, SET *set, int m);
double set_avg_cond_size(XCSF *xcsf, SET *set);
double set_avg_pred_size(XCSF *xcsf, SET *set);
size_t pop_save(XCSF *xcsf, FILE *fp);
size_t pop_load(XCSF *xcsf, FILE *fp);
double set_avg_eta(XCSF *xcsf, SET *set, int layer);
