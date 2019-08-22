/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
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

typedef struct BPN {
    int num_layers; // hidden + output
    int num_inputs;
    int num_outputs;
    struct LAYER *layers;
} BPN;

double neural_output(XCSF *xcsf, BPN *bpn, int i);
void neural_copy(XCSF *xcsf, BPN *to, BPN *from);
void neural_free(XCSF *xcsf, BPN *bpn);
void neural_learn(XCSF *xcsf, BPN *bpn, double *output, double *input);
void neural_print(XCSF *xcsf, BPN *bpn, _Bool print_weights);
void neural_propagate(XCSF *xcsf, BPN *bpn, double *input);
void neural_rand(XCSF *xcsf, BPN *bpn);
void neural_init(XCSF *xcsf, BPN *bpn, int num_layers, int *neurons, int *activations);
_Bool neural_mutate(XCSF *xcsf, BPN *bpn);
