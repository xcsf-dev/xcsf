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
 * @file act_integer.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief integer action functions.
 */

#include "act_integer.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of integer action mutation rates
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL }; //<! Self-adaptation method

_Bool
act_integer_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
act_integer_general(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2)
{
    (void) xcsf;
    const struct ActInteger *act1 = c1->act;
    const struct ActInteger *act2 = c2->act;
    if (act1->action != act2->action) {
        return false;
    }
    return true;
}

_Bool
act_integer_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    struct ActInteger *act = c->act;
    sam_adapt(act->mu, N_MU, MU_TYPE);
    if (rand_uniform(0, 1) < act->mu[0]) {
        const int old = act->action;
        act->action = irand_uniform(0, xcsf->n_actions);
        if (old != act->action) {
            return true;
        }
    }
    return false;
}

int
act_integer_compute(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    (void) xcsf;
    (void) x;
    const struct ActInteger *act = c->act;
    return act->action;
}

void
act_integer_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct ActInteger *new = malloc(sizeof(struct ActInteger));
    const struct ActInteger *src_act = src->act;
    new->action = src_act->action;
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->mu, src_act->mu, sizeof(double) * N_MU);
    dest->act = new;
}

void
act_integer_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActInteger *act = c->act;
    printf("%d\n", act->action);
}

void
act_integer_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const int action)
{
    (void) xcsf;
    (void) x;
    struct ActInteger *act = c->act;
    act->action = action;
}

void
act_integer_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActInteger *act = c->act;
    free(act->mu);
    free(c->act);
}

void
act_integer_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct ActInteger *new = malloc(sizeof(struct ActInteger));
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    new->action = irand_uniform(0, xcsf->n_actions);
    c->act = new;
}

void
act_integer_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

size_t
act_integer_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    const struct ActInteger *act = c->act;
    s += fwrite(&act->action, sizeof(int), 1, fp);
    s += fwrite(act->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t
act_integer_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    struct ActInteger *new = malloc(sizeof(struct ActInteger));
    s += fread(&new->action, sizeof(int), 1, fp);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->act = new;
    return s;
}
