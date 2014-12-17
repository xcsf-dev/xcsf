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
 *
 **************
 * Description: 
 **************
 * The self-adaptive mutation module.
 *
 * Initialises the classifier mutation rates and performs self-adaptation using
 * a normal distribution.
 */

#ifdef SELF_ADAPT_MUTATION   	
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "cons.h"
#include "random.h"
#include "cl.h"

double gasdev(CL *c, int m);

void sam_init(CL *c)
{
	c->mu = malloc(sizeof(double)*NUM_MU);
	c->iset = malloc(sizeof(int)*NUM_MU);
	c->gset = malloc(sizeof(double)*NUM_MU);
	for(int i = 0; i < NUM_MU; i++) {
		c->mu[i] = drand();
		c->iset[i] = 0;
		c->gset[i] = 0.0;
	}     
}

void sam_copy(CL *to, CL *from)
{
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
	memcpy(to->gset, from->gset, sizeof(double)*NUM_MU);
	memcpy(to->iset, from->iset, sizeof(int)*NUM_MU);
}

void sam_free(CL *c)
{
	free(c->mu);
	free(c->iset);
	free(c->gset);
}

void sam_adapt(CL *c)
{
	for(int i = 0; i < NUM_MU; i++) {
		c->mu[i] *= exp(gasdev(c,i));
		if(c->mu[i] < muEPS_0)
			c->mu[i] = muEPS_0;
		else if(c->mu[i] > 1.0)
			c->mu[i] = 1.0;
	}
}

double gasdev(CL *c, int m)
{
	// from numerical recipes in c
	double fac, rsq, v1, v2;
	if(c->iset[m] == 0) {
		do {
			v1 = (drand()*2.0)-1.0;
			v2 = (drand()*2.0)-1.0;
			rsq = (v1*v1)+(v2*v2);
		}
		while(rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0*log(rsq)/rsq);
		c->gset[m] = v1*fac;
		c->iset[m] = 1;
		return v2*fac;
	}
	else {
		c->iset[m] = 0;
		return c->gset[m];
	}
}
#endif     
