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
 * @file cond_dummy.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Always-matching dummy condition functions.
 */

#include "cond_dummy.h"

void
cond_dummy_init(const struct XCSF *xcsf, struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
cond_dummy_free(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
cond_dummy_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

void
cond_dummy_cover(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

void
cond_dummy_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
                  const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
cond_dummy_match(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
    return true;
}

_Bool
cond_dummy_crossover(const struct XCSF *xcsf, const struct CL *c1,
                     const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
cond_dummy_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

_Bool
cond_dummy_general(const struct XCSF *xcsf, const struct CL *c1,
                   const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_dummy_print(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
}

int
cond_dummy_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
    return 0;
}

size_t
cond_dummy_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

size_t
cond_dummy_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}
