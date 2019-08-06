/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
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
 * The random number generator interface module.
 *
 * Initialises the Mersenne Twister random number generator and provides
 * abstracted functions for calculating a random floating point or integer. 
 */

#include <time.h>
#include <limits.h>
#include "mt64.h"
#include "random.h"

void random_init()
{
	time_t now = time (0);
	unsigned char *p = (unsigned char *)&now;
	unsigned seed = 0;
	size_t i;

	for(i = 0; i < sizeof(now); i++)
		seed = (seed * (UCHAR_MAX + 2U)) + p[i];

	init_genrand64(seed);
}

// not inclusive of max
int irand(int min, int max)
{
	return min + (drand() * (max-min));
}

double drand()
{
	// Mersenne Twister 64bit version
	return genrand64_real1();
}
