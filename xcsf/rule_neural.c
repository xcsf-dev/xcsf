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
 * @file rule_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Neural network rule (condition + action) functions.
 */

#include "rule_neural.h"
#include "neural_activations.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_lstm.h"
#include "neural_layer_recurrent.h"
#include "utils.h"

/* CONDITION FUNCTIONS */

void
rule_neural_cond_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct RuleNeural *new = malloc(sizeof(struct RuleNeural));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = neural_cond_lopt(xcsf);
    struct Layer *l = NULL;
    int n_inputs = xcsf->x_dim;
    for (int i = 0; i < MAX_LAYERS && xcsf->COND_NUM_NEURONS[i] > 0; ++i) {
        const int hinit = xcsf->COND_NUM_NEURONS[i];
        int hmax = xcsf->COND_MAX_NEURONS[i];
        if (hmax < hinit || !xcsf->COND_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        const int f = xcsf->COND_HIDDEN_ACTIVATION;
        l = neural_layer_recurrent_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_push(xcsf, &new->net, l);
        n_inputs = hinit;
    }
    // output layer
    const int f = xcsf->COND_OUTPUT_ACTIVATION;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    const int n =
        (int) fmax(1, ceil(log2(xcsf->n_actions))); // n action neurons
    new->n_outputs = n;
    l = neural_layer_connected_init(xcsf, n_inputs, n + 1, n + 1, f, lopt);
    neural_push(xcsf, &new->net, l);
    c->cond = new;
}

void
rule_neural_cond_free(const struct XCSF *xcsf, const struct Cl *c)
{
    struct RuleNeural *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void
rule_neural_cond_copy(const struct XCSF *xcsf, struct Cl *dest,
                      const struct Cl *src)
{
    struct RuleNeural *new = malloc(sizeof(struct RuleNeural));
    const struct RuleNeural *src_cond = src->cond;
    new->n_outputs = src_cond->n_outputs;
    neural_copy(xcsf, &new->net, &src_cond->net);
    dest->cond = new;
}

void
rule_neural_cond_cover(const struct XCSF *xcsf, const struct Cl *c,
                       const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

void
rule_neural_cond_update(const struct XCSF *xcsf, const struct Cl *c,
                        const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
rule_neural_cond_match(const struct XCSF *xcsf, const struct Cl *c,
                       const double *x)
{
    const struct RuleNeural *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if (neural_output(xcsf, &cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

_Bool
rule_neural_cond_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct RuleNeural *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool
rule_neural_cond_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                           const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
rule_neural_cond_general(const struct XCSF *xcsf, const struct Cl *c1,
                         const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
rule_neural_cond_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct RuleNeural *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}

double
rule_neural_cond_size(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct RuleNeural *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t
rule_neural_cond_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct RuleNeural *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    return s;
}

size_t
rule_neural_cond_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct RuleNeural *new = malloc(sizeof(struct RuleNeural));
    size_t s = neural_load(xcsf, &new->net, fp);
    new->n_outputs = (int) fmax(1, ceil(log2(xcsf->n_actions)));
    c->cond = new;
    return s;
}

/* ACTION FUNCTIONS */

void
rule_neural_act_init(const struct XCSF *xcsf, struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_neural_act_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_neural_act_copy(const struct XCSF *xcsf, struct Cl *dest,
                     const struct Cl *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

void
rule_neural_act_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_neural_act_cover(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x, const int action)
{
    const struct RuleNeural *cond = c->cond;
    do {
        neural_rand(xcsf, &cond->net);
    } while (!rule_neural_cond_match(xcsf, c, x) &&
             rule_neural_act_compute(xcsf, c, x) != action);
}

int
rule_neural_act_compute(const struct XCSF *xcsf, const struct Cl *c,
                        const double *x)
{
    (void) x;
    const struct RuleNeural *cond = c->cond;
    int action = 0;
    for (int i = 0; i < cond->n_outputs; ++i) {
        if (neural_output(xcsf, &cond->net, i + 1) > 0.5) {
            action += (int) pow(2, i);
        }
    }
    action = clamp_int(action, 0, xcsf->n_actions - 1);
    return action;
}

void
rule_neural_act_update(const struct XCSF *xcsf, const struct Cl *c,
                       const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
rule_neural_act_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                          const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
rule_neural_act_general(const struct XCSF *xcsf, const struct Cl *c1,
                        const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
rule_neural_act_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

size_t
rule_neural_act_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

size_t
rule_neural_act_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}
