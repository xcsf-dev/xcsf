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

double *pred_nlms_pre(XCSF *xcsf, CL *c);
double *pred_nlms_compute(XCSF *xcsf, CL *c, double *x);
void pred_nlms_copy(XCSF *xcsf, CL *to,  CL *from);
void pred_nlms_free(XCSF *xcsf, CL *c);
void pred_nlms_init(XCSF *xcsf, CL *c);
void pred_nlms_print(XCSF *xcsf, CL *c);
void pred_nlms_update(XCSF *xcsf, CL *c, double *x, double *y);

static struct PredVtbl const pred_nlms_vtbl = {
	&pred_nlms_compute,
	&pred_nlms_pre,
	&pred_nlms_copy,
	&pred_nlms_free,
	&pred_nlms_init,
	&pred_nlms_print,
	&pred_nlms_update
};
