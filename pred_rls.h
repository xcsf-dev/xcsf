/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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

#ifdef RLS_PREDICTION

typedef struct PRED {
	int weights_length;
	double *weights;
	double *matrix;
	double pre;
	// to enable parallel update each temp array must be private
	double *tmp_input;
	double *tmp_vec;
	double *tmp_matrix1;
	double *tmp_matrix2;
} PRED;

#endif
