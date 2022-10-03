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
 * @date 2019--2022.
 * @brief Dynamical GP graph rule (condition + action) functions.
 */

#include "rule_dgp.h"
#include "utils.h"

/* CONDITION FUNCTIONS */

void
rule_dgp_cond_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct RuleDGP *new = malloc(sizeof(struct RuleDGP));
    new->n_outputs = (int) fmax(1, ceil(log2(xcsf->n_actions)));
    graph_init(&new->dgp, xcsf->cond->dargs);
    graph_rand(&new->dgp);
    c->cond = new;
}

void
rule_dgp_cond_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct RuleDGP *cond = c->cond;
    graph_free(&cond->dgp);
    free(c->cond);
}

void
rule_dgp_cond_copy(const struct XCSF *xcsf, struct Cl *dest,
                   const struct Cl *src)
{
    struct RuleDGP *new = malloc(sizeof(struct RuleDGP));
    const struct RuleDGP *src_cond = src->cond;
    graph_init(&new->dgp, xcsf->cond->dargs);
    graph_copy(&new->dgp, &src_cond->dgp);
    new->n_outputs = src_cond->n_outputs;
    dest->cond = new;
}

void
rule_dgp_cond_cover(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

void
rule_dgp_cond_update(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

bool
rule_dgp_cond_match(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    const struct RuleDGP *cond = c->cond;
    graph_update(&cond->dgp, x, !xcsf->STATEFUL);
    if (graph_output(&cond->dgp, 0) > 0.5) {
        return true;
    }
    return false;
}

bool
rule_dgp_cond_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct RuleDGP *cond = c->cond;
    return graph_mutate(&cond->dgp);
}

bool
rule_dgp_cond_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                        const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
rule_dgp_cond_general(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
rule_dgp_cond_print(const struct XCSF *xcsf, const struct Cl *c)
{
    char *json_str = rule_dgp_cond_json_export(xcsf, c);
    printf("%s\n", json_str);
    free(json_str);
}

double
rule_dgp_cond_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct RuleDGP *cond = c->cond;
    return cond->dgp.n;
}

size_t
rule_dgp_cond_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct RuleDGP *cond = c->cond;
    size_t s = graph_save(&cond->dgp, fp);
    return s;
}

size_t
rule_dgp_cond_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct RuleDGP *new = malloc(sizeof(struct RuleDGP));
    size_t s = graph_load(&new->dgp, fp);
    new->n_outputs = (int) fmax(1, ceil(log2(xcsf->n_actions)));
    c->cond = new;
    return s;
}

char *
rule_dgp_cond_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct RuleDGP *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "rule_dgp");
    char *graph_str = graph_json_export(&cond->dgp);
    cJSON *graph = cJSON_Parse(graph_str);
    free(graph_str);
    cJSON_AddItemToObject(json, "graph", graph);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

void
rule_dgp_cond_json_import(const struct XCSF *xcsf, struct Cl *c,
                          const cJSON *json)
{
    const cJSON *item = cJSON_GetObjectItem(json, "graph");
    if (item == NULL) {
        printf("Import error: missing graph\n");
        exit(EXIT_FAILURE);
    }
    struct RuleDGP *cond = c->cond;
    graph_free(&cond->dgp);
    graph_json_import(&cond->dgp, xcsf->cond->dargs, item);
}

/* ACTION FUNCTIONS */

void
rule_dgp_act_init(const struct XCSF *xcsf, struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_dgp_act_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

void
rule_dgp_act_copy(const struct XCSF *xcsf, struct Cl *dest,
                  const struct Cl *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

void
rule_dgp_act_print(const struct XCSF *xcsf, const struct Cl *c)
{
    char *json_str = rule_dgp_act_json_export(xcsf, c);
    printf("%s\n", json_str);
    free(json_str);
}

void
rule_dgp_act_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const int action)
{
    struct RuleDGP *cond = c->cond;
    do {
        graph_rand(&cond->dgp);
    } while (!rule_dgp_cond_match(xcsf, c, x) &&
             rule_dgp_act_compute(xcsf, c, x) != action);
}

int
rule_dgp_act_compute(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    (void) xcsf;
    (void) x;
    const struct RuleDGP *cond = c->cond;
    int action = 0;
    for (int i = 0; i < cond->n_outputs; ++i) {
        if (graph_output(&cond->dgp, i + 1) > 0.5) {
            action += (int) pow(2, i);
        }
    }
    action = clamp_int(action, 0, xcsf->n_actions - 1);
    return action;
}

void
rule_dgp_act_update(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

bool
rule_dgp_act_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
rule_dgp_act_general(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
rule_dgp_act_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

size_t
rule_dgp_act_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

size_t
rule_dgp_act_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

char *
rule_dgp_act_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "rule_dgp");
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

void
rule_dgp_act_json_import(const struct XCSF *xcsf, struct Cl *c,
                         const cJSON *json)
{
    (void) xcsf;
    (void) c;
    (void) json;
}
