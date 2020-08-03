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
 * @file rule_dgp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Dynamical GP graph rule (condition + action) functions.
 */

#include "rule_dgp.h"
#include "utils.h"

/* CONDITION FUNCTIONS */

void
rule_dgp_cond_init(const struct XCSF *xcsf, struct CL *c)
{
    struct RULE_DGP *new = malloc(sizeof(struct RULE_DGP));
    new->n_outputs = (int) fmax(1, ceil(log2(xcsf->n_actions)));
    int n = (int) fmax(xcsf->COND_NUM_NEURONS[0], new->n_outputs + 1);
    graph_init(xcsf, &new->dgp, n);
    graph_rand(xcsf, &new->dgp);
    c->cond = new;
}

void
rule_dgp_cond_free(const struct XCSF *xcsf, const struct CL *c)
{
    const struct RULE_DGP *cond = c->cond;
    graph_free(xcsf, &cond->dgp);
    free(c->cond);
}

void
rule_dgp_cond_copy(const struct XCSF *xcsf, struct CL *dest,
                   const struct CL *src)
{
    struct RULE_DGP *new = malloc(sizeof(struct RULE_DGP));
    const struct RULE_DGP *src_cond = src->cond;
    graph_init(xcsf, &new->dgp, src_cond->dgp.n);
    graph_copy(xcsf, &new->dgp, &src_cond->dgp);
    new->n_outputs = src_cond->n_outputs;
    dest->cond = new;
}

void
rule_dgp_cond_cover(const struct XCSF *xcsf, const struct CL *c,
                    const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

void
rule_dgp_cond_update(const struct XCSF *xcsf, const struct CL *c,
                     const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
rule_dgp_cond_match(const struct XCSF *xcsf, const struct CL *c,
                    const double *x)
{
    const struct RULE_DGP *cond = c->cond;
    graph_update(xcsf, &cond->dgp, x);
    if (graph_output(xcsf, &cond->dgp, 0) > 0.5) {
        return true;
    }
    return false;
}

_Bool
rule_dgp_cond_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    struct RULE_DGP *cond = c->cond;
    return graph_mutate(xcsf, &cond->dgp);
}

_Bool
rule_dgp_cond_crossover(const struct XCSF *xcsf, const struct CL *c1,
                        const struct CL *c2)
{
    struct RULE_DGP *cond1 = c1->cond;
    struct RULE_DGP *cond2 = c2->cond;
    return graph_crossover(xcsf, &cond1->dgp, &cond2->dgp);
}

_Bool
rule_dgp_cond_general(const struct XCSF *xcsf, const struct CL *c1,
                      const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
rule_dgp_cond_print(const struct XCSF *xcsf, const struct CL *c)
{
    const struct RULE_DGP *cond = c->cond;
    graph_print(xcsf, &cond->dgp);
}

int
rule_dgp_cond_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    const struct RULE_DGP *cond = c->cond;
    return cond->dgp.n;
}

size_t
rule_dgp_cond_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    const struct RULE_DGP *cond = c->cond;
    size_t s = graph_save(xcsf, &cond->dgp, fp);
    return s;
}

size_t
rule_dgp_cond_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    struct RULE_DGP *new = malloc(sizeof(struct RULE_DGP));
    size_t s = graph_load(xcsf, &new->dgp, fp);
    new->n_outputs = (int) fmax(1, ceil(log2(xcsf->n_actions)));
    c->cond = new;
    return s;
}

/* ACTION FUNCTIONS */

void
rule_dgp_act_init(const struct XCSF *xcsf, struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_dgp_act_free(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_dgp_act_copy(const struct XCSF *xcsf, struct CL *dest,
                  const struct CL *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

void
rule_dgp_act_print(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_dgp_act_cover(const struct XCSF *xcsf, const struct CL *c, const double *x,
                   int action)
{
    struct RULE_DGP *cond = c->cond;
    do {
        graph_rand(xcsf, &cond->dgp);
    } while (!rule_dgp_cond_match(xcsf, c, x) &&
             rule_dgp_act_compute(xcsf, c, x) != action);
}

int
rule_dgp_act_compute(const struct XCSF *xcsf, const struct CL *c,
                     const double *x)
{
    (void) x;
    const struct RULE_DGP *cond = c->cond;
    int action = 0;
    for (int i = 0; i < cond->n_outputs; ++i) {
        if (graph_output(xcsf, &cond->dgp, i + 1) > 0.5) {
            action += (int) pow(2, i);
        }
    }
    action = iclamp(action, 0, xcsf->n_actions - 1);
    return action;
}

void
rule_dgp_act_update(const struct XCSF *xcsf, const struct CL *c,
                    const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
rule_dgp_act_crossover(const struct XCSF *xcsf, const struct CL *c1,
                       const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
rule_dgp_act_general(const struct XCSF *xcsf, const struct CL *c1,
                     const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
rule_dgp_act_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

size_t
rule_dgp_act_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

size_t
rule_dgp_act_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}
