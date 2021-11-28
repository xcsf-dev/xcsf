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
 * @date 2016--2021.
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
    graph_init(&new->dgp, xcsf->cond->dargs);
    graph_rand(&new->dgp);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a dynamical GP graph condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_dgp_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    graph_free(&cond->dgp);
    free(c->cond);
}

/**
 * @brief Copies a dynamical GP graph condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_dgp_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct CondDGP *new = malloc(sizeof(struct CondDGP));
    const struct CondDGP *src_cond = src->cond;
    graph_init(&new->dgp, xcsf->cond->dargs);
    graph_copy(&new->dgp, &src_cond->dgp);
    dest->cond = new;
}

/**
 * @brief Generates a dynamical GP graph that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_dgp_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    (void) xcsf;
    struct CondDGP *cond = c->cond;
    do {
        graph_rand(&cond->dgp);
    } while (!cond_dgp_match(xcsf, c, x));
}

/**
 * @brief Dummy update function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_dgp_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Calculates whether a dynamical GP graph condition matches an input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return Whether the dynamical GP graph condition matches the input.
 */
bool
cond_dgp_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondDGP *cond = c->cond;
    graph_update(&cond->dgp, x, !xcsf->STATEFUL);
    if (graph_output(&cond->dgp, 0) > 0.5) {
        return true;
    }
    return false;
}

/**
 * @brief Mutates a dynamical GP graph condition with the self-adaptive rates.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_dgp_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct CondDGP *cond = c->cond;
    return graph_mutate(&cond->dgp);
}

/**
 * @brief Dummy crossover function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return False.
 */
bool
cond_dgp_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy general function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return False.
 */
bool
cond_dgp_general(const struct XCSF *xcsf, const struct Cl *c1,
                 const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Prints a dynamical GP graph condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_dgp_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_dgp_json_export(xcsf, c));
}

/**
 * @brief Returns the size of a dynamical GP graph condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The number of nodes in the graph.
 */
double
cond_dgp_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    return cond->dgp.n;
}

/**
 * @brief Writes a dynamical GP graph condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_dgp_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    size_t s = graph_save(&cond->dgp, fp);
    return s;
}

/**
 * @brief Reads a dynamical GP graph condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_dgp_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct CondDGP *new = malloc(sizeof(struct CondDGP));
    size_t s = graph_load(&new->dgp, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a DGP condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_dgp_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondDGP *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "dgp");
    cJSON *graph = cJSON_Parse(graph_json_export(&cond->dgp));
    cJSON_AddItemToObject(json, "graph", graph);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
