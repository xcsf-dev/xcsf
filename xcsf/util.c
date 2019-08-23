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
 */

#include <time.h>
#include <limits.h>
#include <math.h>
#include "mt64.h"
#include "util.h"

void random_init()
{
	time_t now = time(0);
	unsigned char *p = (unsigned char *)&now;
	unsigned seed = 0;
	for(size_t i = 0; i < sizeof(now); i++) {
		seed = (seed * (UCHAR_MAX + 2U)) + p[i];
	}
	init_genrand64(seed);
}
 
double drand()
{
	// Mersenne Twister 64bit version
	return genrand64_real1();
}
 
int irand_uniform(int min, int max)
{
    // not inclusive of max
	return min + (drand() * (max-min));
}

double rand_uniform(double min, double max)
{
	return min + (drand() * (max-min));
}

double rand_normal()
{
    // from numerical recipes in c
    static int iset = 0;
    static double gset;
    if(iset == 0) {
        double fac, rsq, v1, v2;
        do {
            v1 = rand_uniform(-1,1);
            v2 = rand_uniform(-1,1);
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

double constrain(double min, double max, double a)
{
    if (a < min) {return min;}
    if (a > max) {return max;}
    return a;
}

int iconstrain(int min, int max, int a)
{
    if (a < min) {return min;}
    if (a > max) {return max;}
    return a;
}
