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
 * @file cond_gp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Tree GP condition functions.
 */

#include "cond_gp.h"
#include "sam.h"
#include "utils.h"

/**
 * @brief Creates and initialises a tree-GP condition.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose condition is to be initialised.
 */
void
cond_gp_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondGP *new = malloc(sizeof(struct CondGP));
    tree_rand(xcsf, &new->gp);
    c->cond = new;
}

void
cond_gp_free(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondGP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    free(c->cond);
}

void
cond_gp_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct CondGP *new = malloc(sizeof(struct CondGP));
    const struct CondGP *src_cond = src->cond;
    tree_copy(xcsf, &new->gp, &src_cond->gp);
    dest->cond = new;
}

void
cond_gp_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondGP *cond = c->cond;
    do {
        tree_free(xcsf, &cond->gp);
        tree_rand(xcsf, &cond->gp);
    } while (!cond_gp_match(xcsf, c, x));
}

void
cond_gp_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
               const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
cond_gp_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondGP *cond = c->cond;
    cond->gp.p = 0;
    if (tree_eval(xcsf, &cond->gp, x) > 0.5) {
        return true;
    }
    return false;
}

_Bool
cond_gp_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    struct CondGP *cond = c->cond;
    return tree_mutate(xcsf, &cond->gp);
}

_Bool
cond_gp_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                  const struct Cl *c2)
{
    struct CondGP *cond1 = c1->cond;
    struct CondGP *cond2 = c2->cond;
    if (rand_uniform(0, 1) < xcsf->P_CROSSOVER) {
        tree_crossover(xcsf, &cond1->gp, &cond2->gp);
        return true;
    }
    return false;
}

_Bool
cond_gp_general(const struct XCSF *xcsf, const struct Cl *c1,
                const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_gp_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondGP *cond = c->cond;
    printf("GP tree: ");
    tree_print(xcsf, &cond->gp, 0);
    printf("\n");
}

double
cond_gp_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondGP *cond = c->cond;
    return cond->gp.len;
}

size_t
cond_gp_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct CondGP *cond = c->cond;
    size_t s = tree_save(xcsf, &cond->gp, fp);
    return s;
}

size_t
cond_gp_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct CondGP *new = malloc(sizeof(struct CondGP));
    size_t s = tree_load(xcsf, &new->gp, fp);
    c->cond = new;
    return s;
}
