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

double set_mean_time(XCSF *xcsf, NODE **set, int num_sum);
double set_total_fit(XCSF *xcsf, NODE **set);
double set_total_time(XCSF *xcsf, NODE **set);
void pop_add(XCSF *xcsf, CL *c);
void pop_del(XCSF *xcsf, NODE **kset);
void pop_enforce_limit(XCSF *xcsf, NODE **kset);
void pop_init(XCSF *xcsf);
void set_add(XCSF *xcsf, NODE **set, CL *c);
void set_free(XCSF *xcsf, NODE **set);
void set_kill(XCSF *xcsf, NODE **set);
void set_match(XCSF *xcsf, NODE **set, int *size, int *num, double *x, int time, NODE **kset);
void set_pred(XCSF *xcsf, NODE **set, int size, double *x, double *y);
void set_print(XCSF *xcsf, NODE *set);
void set_times(XCSF *xcsf, NODE **set, int time);
void set_update(XCSF *xcsf, NODE **set, int *size, int *num, double *y, NODE **kset, double *x);
void set_validate(XCSF *xcsf, NODE **set, int *size, int *num);
double set_avg_mut(XCSF *xcsf, NODE **set, int m);
