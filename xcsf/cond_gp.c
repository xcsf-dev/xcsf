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
cond_gp_init(const struct XCSF *xcsf, struct CL *c)
{
    struct COND_GP *new = malloc(sizeof(struct COND_GP));
    tree_rand(xcsf, &new->gp);
    c->cond = new;
}

void
cond_gp_free(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_GP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    free(c->cond);
}

void
cond_gp_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src)
{
    struct COND_GP *new = malloc(sizeof(struct COND_GP));
    const struct COND_GP *src_cond = src->cond;
    tree_copy(xcsf, &new->gp, &src_cond->gp);
    dest->cond = new;
}

void
cond_gp_cover(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    struct COND_GP *cond = c->cond;
    do {
        tree_free(xcsf, &cond->gp);
        tree_rand(xcsf, &cond->gp);
    } while (!cond_gp_match(xcsf, c, x));
}

void
cond_gp_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
               const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
cond_gp_match(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    struct COND_GP *cond = c->cond;
    cond->gp.p = 0;
    double result = tree_eval(xcsf, &cond->gp, x);
    if (result > 0.5) {
        return true;
    }
    return false;
}

_Bool
cond_gp_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    struct COND_GP *cond = c->cond;
    return tree_mutate(xcsf, &cond->gp);
}

_Bool
cond_gp_crossover(const struct XCSF *xcsf, const struct CL *c1,
                  const struct CL *c2)
{
    struct COND_GP *cond1 = c1->cond;
    struct COND_GP *cond2 = c2->cond;
    if (rand_uniform(0, 1) < xcsf->P_CROSSOVER) {
        tree_crossover(xcsf, &cond1->gp, &cond2->gp);
        return true;
    }
    return false;
}

_Bool
cond_gp_general(const struct XCSF *xcsf, const struct CL *c1,
                const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_gp_print(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_GP *cond = c->cond;
    printf("GP tree: ");
    tree_print(xcsf, &cond->gp, 0);
    printf("\n");
}

double
cond_gp_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    const struct COND_GP *cond = c->cond;
    return cond->gp.len;
}

size_t
cond_gp_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    const struct COND_GP *cond = c->cond;
    size_t s = tree_save(xcsf, &cond->gp, fp);
    return s;
}

size_t
cond_gp_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    struct COND_GP *new = malloc(sizeof(struct COND_GP));
    size_t s = tree_load(xcsf, &new->gp, fp);
    c->cond = new;
    return s;
}
