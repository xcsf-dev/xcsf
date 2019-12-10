/*
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
                 
/**
 * @file utils.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Utility functions for random number handling, etc.
 */ 

#pragma once
 
double rand_normal(double mu, double sigma);
double constrain(double min, double max, double a);
double rand_uniform(double min, double max);
int iconstrain(int min, int max, int a);
int irand_uniform(int min, int max);
void random_init();
void float_to_binary(double f, char *binary, int bits);
