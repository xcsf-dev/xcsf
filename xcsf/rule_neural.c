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
 * @date 2019--2021.
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
    neural_create(&new->net, xcsf->cond->largs);
    const int expected = (int) fmax(1, ceil(log2(xcsf->n_actions))) + 1;
    if (new->net.n_outputs != expected) {
        printf("rule_neural_init(): n_outputs(%d) != expected(%d)\n",
               new->net.n_outputs, expected);
        printf("neural rules output binary actions + 1 matching neuron\n");
        exit(EXIT_FAILURE);
    }
    c->cond = new;
}

void
rule_neural_cond_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct RuleNeural *cond = c->cond;
    neural_free(&cond->net);
    free(c->cond);
}

void
rule_neural_cond_copy(const struct XCSF *xcsf, struct Cl *dest,
                      const struct Cl *src)
{
    (void) xcsf;
    struct RuleNeural *new = malloc(sizeof(struct RuleNeural));
    const struct RuleNeural *src_cond = src->cond;
    neural_copy(&new->net, &src_cond->net);
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

bool
rule_neural_cond_match(const struct XCSF *xcsf, const struct Cl *c,
                       const double *x)
{
    struct RuleNeural *cond = c->cond;
    neural_propagate(&cond->net, x, xcsf->explore);
    if (neural_output(&cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

bool
rule_neural_cond_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct RuleNeural *cond = c->cond;
    return neural_mutate(&cond->net);
}

bool
rule_neural_cond_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                           const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
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
    printf("%s\n", rule_neural_cond_json_export(xcsf, c));
}

double
rule_neural_cond_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct RuleNeural *cond = c->cond;
    return neural_size(&cond->net);
}

size_t
rule_neural_cond_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct RuleNeural *cond = c->cond;
    size_t s = neural_save(&cond->net, fp);
    return s;
}

size_t
rule_neural_cond_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct RuleNeural *new = malloc(sizeof(struct RuleNeural));
    size_t s = neural_load(&new->net, fp);
    c->cond = new;
    return s;
}

const char *
rule_neural_cond_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    const struct RuleNeural *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "rule_neural");
    cJSON *network = cJSON_Parse(neural_json_export(&cond->net, false));
    cJSON_AddItemToObject(json, "network", network);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
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
    printf("%s\n", rule_neural_act_json_export(xcsf, c));
}

void
rule_neural_act_cover(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x, const int action)
{
    const struct RuleNeural *cond = c->cond;
    do {
        neural_rand(&cond->net);
    } while (!rule_neural_cond_match(xcsf, c, x) &&
             rule_neural_act_compute(xcsf, c, x) != action);
}

int
rule_neural_act_compute(const struct XCSF *xcsf, const struct Cl *c,
                        const double *x)
{
    (void) xcsf;
    (void) x;
    const struct RuleNeural *cond = c->cond;
    int action = 0;
    for (int i = 1; i < cond->net.n_outputs; ++i) {
        if (neural_output(&cond->net, i) > 0.5) {
            action += (int) pow(2, i - 1);
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

bool
rule_neural_act_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                          const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
rule_neural_act_general(const struct XCSF *xcsf, const struct Cl *c1,
                        const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
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

const char *
rule_neural_act_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "rule_neural");
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
