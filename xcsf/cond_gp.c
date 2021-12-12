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
 * @date 2016--2021.
 * @brief Tree GP condition functions.
 */

#include "cond_gp.h"
#include "ea.h"
#include "sam.h"
#include "utils.h"

/**
 * @brief Creates and initialises a tree-GP condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
void
cond_gp_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondGP *new = malloc(sizeof(struct CondGP));
    tree_rand(&new->gp, xcsf->cond->targs);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a tree-GP condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_gp_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondGP *cond = c->cond;
    tree_free(&cond->gp);
    free(c->cond);
}

/**
 * @brief Copies a tree-GP condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_gp_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct CondGP *new = malloc(sizeof(struct CondGP));
    const struct CondGP *src_cond = src->cond;
    tree_copy(&new->gp, &src_cond->gp);
    dest->cond = new;
}

/**
 * @brief Generates a GP tree that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_gp_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondGP *cond = c->cond;
    do {
        tree_free(&cond->gp);
        tree_rand(&cond->gp, xcsf->cond->targs);
    } while (!cond_gp_match(xcsf, c, x));
}

/**
 * @brief Dummy update function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_gp_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
               const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Calculates whether a GP tree condition matches an input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return Whether the hyperrectangle condition matches the input.
 */
bool
cond_gp_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondGP *cond = c->cond;
    cond->gp.pos = 0;
    if (tree_eval(&cond->gp, xcsf->cond->targs, x) > 0.5) {
        return true;
    }
    return false;
}

/**
 * @brief Mutates a tree-GP condition with the self-adaptive rate.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_gp_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct CondGP *cond = c->cond;
    return tree_mutate(&cond->gp, xcsf->cond->targs);
}

/**
 * @brief Performs sub-tree crossover with two tree-GP conditions.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
bool
cond_gp_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                  const struct Cl *c2)
{
    (void) xcsf;
    struct CondGP *cond1 = c1->cond;
    struct CondGP *cond2 = c2->cond;
    if (rand_uniform(0, 1) < xcsf->ea->p_crossover) {
        tree_crossover(&cond1->gp, &cond2->gp);
        return true;
    }
    return false;
}

/**
 * @brief Dummy general function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return False
 */
bool
cond_gp_general(const struct XCSF *xcsf, const struct Cl *c1,
                const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Prints a tree-GP condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_gp_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_gp_json_export(xcsf, c));
}

/**
 * @brief Returns the size of a tree-GP condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The length of the tree.
 */
double
cond_gp_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondGP *cond = c->cond;
    return cond->gp.len;
}

/**
 * @brief Writes a tree-GP condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_gp_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct CondGP *cond = c->cond;
    size_t s = tree_save(&cond->gp, fp);
    return s;
}

/**
 * @brief Reads a tree-GP condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_gp_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct CondGP *new = malloc(sizeof(struct CondGP));
    size_t s = tree_load(&new->gp, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a tree-GP condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_gp_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondGP *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "tree_gp");
    cJSON *tree = cJSON_Parse(tree_json_export(&cond->gp, xcsf->cond->targs));
    cJSON_AddItemToObject(json, "tree", tree);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
