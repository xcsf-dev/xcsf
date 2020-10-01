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
 * @file cond_dgp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Dynamical GP graph condition functions.
 */

#include "cond_dgp.h"
#include "sam.h"
#include "utils.h"

/**
 * @brief Creates and initialises a dynamical GP graph condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
void
cond_dgp_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondDGP *new = malloc(sizeof(struct CondDGP));
    const int n = (int) fmax(xcsf->COND_NUM_NEURONS[0], 1);
    graph_init(xcsf, &new->dgp, n);
    graph_rand(xcsf, &new->dgp);
    c->cond = new;
}

void
cond_dgp_free(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondDGP *cond = c->cond;
    graph_free(xcsf, &cond->dgp);
    free(c->cond);
}

void
cond_dgp_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct CondDGP *new = malloc(sizeof(struct CondDGP));
    const struct CondDGP *src_cond = src->cond;
    graph_init(xcsf, &new->dgp, src_cond->dgp.n);
    graph_copy(xcsf, &new->dgp, &src_cond->dgp);
    dest->cond = new;
}

void
cond_dgp_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondDGP *cond = c->cond;
    do {
        graph_rand(xcsf, &cond->dgp);
    } while (!cond_dgp_match(xcsf, c, x));
}

void
cond_dgp_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
cond_dgp_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondDGP *cond = c->cond;
    graph_update(xcsf, &cond->dgp, x);
    if (graph_output(xcsf, &cond->dgp, 0) > 0.5) {
        return true;
    }
    return false;
}

_Bool
cond_dgp_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    struct CondDGP *cond = c->cond;
    return graph_mutate(xcsf, &cond->dgp);
}

_Bool
cond_dgp_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    struct CondDGP *cond1 = c1->cond;
    struct CondDGP *cond2 = c2->cond;
    return graph_crossover(xcsf, &cond1->dgp, &cond2->dgp);
}

_Bool
cond_dgp_general(const struct XCSF *xcsf, const struct Cl *c1,
                 const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_dgp_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    graph_print(xcsf, &cond->dgp);
}

double
cond_dgp_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    return cond->dgp.n;
}

size_t
cond_dgp_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct CondDGP *cond = c->cond;
    size_t s = graph_save(xcsf, &cond->dgp, fp);
    return s;
}

size_t
cond_dgp_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct CondDGP *new = malloc(sizeof(struct CondDGP));
    size_t s = graph_load(xcsf, &new->dgp, fp);
    c->cond = new;
    return s;
}
