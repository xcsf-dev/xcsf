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

double *pred_rls_compute(XCSF *xcsf, CL *c, double *x);
void pred_rls_copy(XCSF *xcsf, CL *to,  CL *from);
void pred_rls_free(XCSF *xcsf, CL *c);
void pred_rls_init(XCSF *xcsf, CL *c);
void pred_rls_print(XCSF *xcsf, CL *c);
void pred_rls_update(XCSF *xcsf, CL *c, double *x, double *y);

static struct PredVtbl const pred_rls_vtbl = {
	&pred_rls_compute,
	&pred_rls_copy,
	&pred_rls_free,
	&pred_rls_init,
	&pred_rls_print,
	&pred_rls_update
};
