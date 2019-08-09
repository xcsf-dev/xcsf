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
 *
 **************
 * Description: 
 **************
 * The self-adaptive mutation module.
 *
 * Initialises the classifier mutation rates and performs self-adaptation using
 * a normal distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "data_structures.h"
#include "random.h"

double gasdev();

void sam_init(XCSF *xcsf, double **mu)
{
	if(xcsf->NUM_SAM > 0) {
		*mu = malloc(sizeof(double) * xcsf->NUM_SAM);
		for(int i = 0; i < xcsf->NUM_SAM; i++) {
			(*mu)[i] = drand();
		}
	}
}

void sam_copy(XCSF *xcsf, double *to, double *from)
{
	memcpy(to, from, sizeof(double) * xcsf->NUM_SAM);
}

void sam_free(XCSF *xcsf, double *mu)
{
	if(xcsf->NUM_SAM > 0) {
		free(mu);
	}
}

void sam_adapt(XCSF *xcsf, double *mu)
{
	for(int i = 0; i < xcsf->NUM_SAM; i++) {
		mu[i] *= exp(gasdev());
		if(mu[i] < xcsf->muEPS_0) {
			mu[i] = xcsf->muEPS_0;
		}
		else if(mu[i] > 1.0) {
			mu[i] = 1.0;
		}
	}
}

void sam_print(XCSF *xcsf, double *mu)
{
	printf("mu: \n");
	for(int i = 0; i < xcsf->NUM_SAM; i++) {
		printf("%f, ", mu[i]);
	}
	printf("\n");
}

double gasdev()
{
	// from numerical recipes in c
	static int iset = 0;
	static double gset;
	double fac, rsq, v1;
	if(iset == 0) {
		double v2;
		do {
			v1 = (drand()*2.0)-1.0;
			v2 = (drand()*2.0)-1.0;
			rsq = (v1*v1)+(v2*v2);
		} while(rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0*log(rsq)/rsq);
		gset = v1*fac;
		iset = 1;
		return v2*fac;
	}
	else {
		iset = 0;
		return gset;
	}
}
