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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "mt64.h"
#include "utils.h"

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
    return floor(rand_uniform(min, max));
}

double rand_uniform(double min, double max)
{
    return min + (drand() * (max-min));
}

double rand_normal(double mu, double sigma)
{
    // Box-Muller transform
    static const double epsilon = DBL_MIN;
    static const double two_pi = 2*M_PI;
    static double z1;
    static _Bool generate;
    generate = !generate;
    if(!generate) {
        return z1 * sigma + mu;
    }
    double u1, u2;
    do {
        u1 = drand();
        u2 = drand();
    } while(u1 <= epsilon);
    double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
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
