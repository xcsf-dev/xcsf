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
 * @date 2015--2021.
 * @brief integer action functions.
 */

#include "act_integer.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of integer action mutation rates

/**
 * @brief Self-adaptation method for mutating integer actions.
 */
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL };

/**
 * @brief Dummy function since integer actions do not perform crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose action is being crossed.
 * @param [in] c2 The second classifier whose action is being crossed.
 * @return False.
 */
bool
act_integer_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Returns whether the action of classifier c1 is more general than c2.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier whose action is tested to be more general.
 * @param [in] c2 The classifier whose action is tested to be more specific.
 * @return Whether the action of c1 is more general than c2.
 */
bool
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

/**
 * @brief Mutates an integer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being mutated.
 * @return Whether any alterations were made.
 */
bool
act_integer_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    struct ActInteger *act = c->act;
    sam_adapt(act->mu, N_MU, MU_TYPE);
    if (rand_uniform(0, 1) < act->mu[0]) {
        const int old = act->action;
        act->action = rand_uniform_int(0, xcsf->n_actions);
        if (old != act->action) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Returns a classifier's integer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action to return.
 * @param [in] x The input state.
 * @return The classifier's action.
 */
int
act_integer_compute(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    (void) xcsf;
    (void) x;
    const struct ActInteger *act = c->act;
    return act->action;
}

/**
 * @brief Copies an integer action from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
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

/**
 * @brief Prints an integer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be printed.
 */
void
act_integer_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", act_integer_json_export(xcsf, c));
}

/**
 * @brief Sets an integer action to a specified value.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being covered.
 * @param [in] x The input state to cover.
 * @param [in] action The action to cover.
 */
void
act_integer_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const int action)
{
    (void) xcsf;
    (void) x;
    struct ActInteger *act = c->act;
    act->action = action;
}

/**
 * @brief Frees the memory used by an integer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be freed.
 */
void
act_integer_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActInteger *act = c->act;
    free(act->mu);
    free(c->act);
}

/**
 * @brief Initialises an integer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be initialised.
 */
void
act_integer_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct ActInteger *new = malloc(sizeof(struct ActInteger));
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    new->action = rand_uniform_int(0, xcsf->n_actions);
    c->act = new;
}

/**
 * @brief Dummy function since integer actions are not updated.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c A classifier data structure.
 * @param [in] x The input state.
 * @param [in] y The payoff value.
 */
void
act_integer_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Writes an integer action to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
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

/**
 * @brief Reads an integer action from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
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

/**
 * @brief Returns a json formatted string representation of an integer action.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose action is to be returned.
 * @return String encoded in json format.
 */
const char *
act_integer_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActInteger *act = c->act;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "integer");
    cJSON_AddNumberToObject(json, "action", act->action);
    cJSON *mutation = cJSON_CreateDoubleArray(act->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
