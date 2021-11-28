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
 * @file cl.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Functions operating on classifiers.
 */

#include "cl.h"
#include "action.h"
#include "condition.h"
#include "ea.h"
#include "loss.h"
#include "prediction.h"
#include "utils.h"

/**
 * @brief Initialises a new classifier - but not condition, action, prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier data structure to initialise.
 * @param [in] size The initial set size value.
 * @param [in] time The current EA time.
 */
void
cl_init(const struct XCSF *xcsf, struct Cl *c, const double size,
        const int time)
{
    c->fit = xcsf->INIT_FITNESS;
    c->err = xcsf->INIT_ERROR;
    c->num = 1;
    c->exp = 0;
    c->size = size;
    c->time = time;
    c->prediction = calloc(xcsf->y_dim, sizeof(double));
    c->action = 0;
    c->m = false;
    c->age = 0;
    c->mtotal = 0;
}

/**
 * @brief Copies condition, action, and prediction structures.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
cl_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    dest->cond_vptr = src->cond_vptr;
    dest->pred_vptr = src->pred_vptr;
    dest->act_vptr = src->act_vptr;
    act_copy(xcsf, dest, src);
    cond_copy(xcsf, dest, src);
    if (xcsf->ea->pred_reset) {
        pred_init(xcsf, dest);
    } else {
        pred_copy(xcsf, dest, src);
    }
}

/**
 * @brief Initialises and creates a copy of one classifier from another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
cl_init_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    dest->prediction = calloc(xcsf->y_dim, sizeof(double));
    dest->fit = src->fit;
    dest->err = src->err;
    dest->num = src->num;
    dest->exp = src->exp;
    dest->size = src->size;
    dest->time = src->time;
    dest->action = src->action;
    dest->m = src->m;
    dest->age = src->age;
    dest->mtotal = src->mtotal;
    dest->cond_vptr = src->cond_vptr;
    dest->pred_vptr = src->pred_vptr;
    dest->act_vptr = src->act_vptr;
    act_copy(xcsf, dest, src);
    cond_copy(xcsf, dest, src);
    pred_copy(xcsf, dest, src);
}

/**
 * @brief Covers the condition and action for a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier being covered.
 * @param [in] x The input state to cover.
 * @param [in] action The action to cover.
 */
void
cl_cover(const struct XCSF *xcsf, struct Cl *c, const double *x,
         const int action)
{
    cl_rand(xcsf, c);
    cond_cover(xcsf, c, x);
    act_cover(xcsf, c, x, action);
    c->m = true;
    c->action = action;
}

/**
 * @brief Initialises random actions, conditions and predictions.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier being randomly initialised.
 */
void
cl_rand(const struct XCSF *xcsf, struct Cl *c)
{
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c);
    cond_init(xcsf, c);
    pred_init(xcsf, c);
    act_init(xcsf, c);
}

/**
 * @brief Returns the deletion vote of a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to calculate the deletion vote.
 * @param [in] avg_fit The population mean fitness.
 * @return The classifier's deletion vote.
 */
double
cl_del_vote(const struct XCSF *xcsf, const struct Cl *c, const double avg_fit)
{
    if (c->exp > xcsf->THETA_DEL && c->fit < xcsf->DELTA * avg_fit * c->num) {
        return c->size * c->num * avg_fit / (c->fit / c->num);
    }
    return c->size * c->num;
}

/**
 * @brief Returns the accuracy of a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier.
 * @return The classifier's accuracy.
 */
double
cl_acc(const struct XCSF *xcsf, const struct Cl *c)
{
    if (c->err > xcsf->E0) {
        const double acc = xcsf->ALPHA * pow(c->err / xcsf->E0, -(xcsf->NU));
        return fmax(acc, DBL_EPSILON);
    }
    return 1;
}

/**
 * @brief Updates a classifier's experience, error, and set size.
 * @details Condition, action, and prediction are updated depending on the
 * knowledge representation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to update.
 * @param [in] x The input state.
 * @param [in] y The true (payoff) value.
 * @param [in] set_num The number of micro-classifiers in the set.
 * @param [in] cur Whether the payoff is for the current or previous state.
 */
void
cl_update(const struct XCSF *xcsf, struct Cl *c, const double *x,
          const double *y, const int set_num, const bool cur)
{
    ++(c->exp);
    if (!cur) { // propagate inputs for the previous state update
        cl_predict(xcsf, c, x);
    }
    const double error = (xcsf->loss_ptr)(xcsf, c->prediction, y);
    if (c->exp * xcsf->BETA < 1) {
        c->err = (c->err * (c->exp - 1) + error) / c->exp;
        c->size = (c->size * (c->exp - 1) + set_num) / c->exp;
    } else {
        c->err += xcsf->BETA * (error - c->err);
        c->size += xcsf->BETA * (set_num - c->size);
    }
    cond_update(xcsf, c, x, y);
    pred_update(xcsf, c, x, y);
    act_update(xcsf, c, x, y);
}

/**
 * @brief Updates the fitness of a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to update.
 * @param [in] acc_sum The sum of all accuracies in the set.
 * @param [in] acc The accuracy of the classifier being updated.
 */
void
cl_update_fit(const struct XCSF *xcsf, struct Cl *c, const double acc_sum,
              const double acc)
{
    c->fit += xcsf->BETA * ((acc * c->num) / acc_sum - c->fit);
}

/**
 * @brief Frees the memory used by a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to free.
 */
void
cl_free(const struct XCSF *xcsf, struct Cl *c)
{
    free(c->prediction);
    cond_free(xcsf, c);
    act_free(xcsf, c);
    pred_free(xcsf, c);
    free(c);
}

/**
 * @brief Prints a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to print.
 * @param [in] print_cond Whether to print the condition.
 * @param [in] print_act Whether to print the action.
 * @param [in] print_pred Whether to print the prediction.
 */
void
cl_print(const struct XCSF *xcsf, const struct Cl *c, const bool print_cond,
         const bool print_act, const bool print_pred)
{
    printf("%s\n", cl_json_export(xcsf, c, print_cond, print_act, print_pred));
}

/**
 * @brief Calculates whether a classifier matches an input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to test for matching.
 * @param [in] x The input state to be matched.
 * @return Whether the classifier matches the input.
 */
bool
cl_match(const struct XCSF *xcsf, struct Cl *c, const double *x)
{
    c->m = cond_match(xcsf, c, x);
    if (c->m) {
        ++(c->mtotal);
    }
    ++(c->age);
    return c->m;
}

/**
 * @brief Returns the fraction of observed inputs matched by a classifier.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier data structure.
 * @return The fraction of matching inputs.
 */
double
cl_mfrac(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    if (c->age > 0) {
        return (double) c->mtotal / c->age;
    }
    return 0;
}

/**
 * @brief Returns whether a classifier matched the most recent input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to test for matching.
 * @return Whether the classifier matched the most recent input.
 */
bool
cl_m(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    return c->m;
}

/**
 * @brief Computes the current classifier action using the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the action.
 * @param [in] x The input state.
 * @return The classifier's action.
 */
int
cl_action(const struct XCSF *xcsf, struct Cl *c, const double *x)
{
    c->action = act_compute(xcsf, c, x);
    return c->action;
}

/**
 * @brief Computes the current classifier prediction using the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier making the prediction.
 * @param [in] x The input state.
 * @return The classifier's (payoff) predictions.
 */
const double *
cl_predict(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    pred_compute(xcsf, c, x);
    return c->prediction;
}

/**
 * @brief Returns whether a classifier is a potential subsumer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to print.
 * @return Whether the classifier is an eligible subsumer.
 */
bool
cl_subsumer(const struct XCSF *xcsf, const struct Cl *c)
{
    if (c->exp > xcsf->THETA_SUB && c->err < xcsf->E0) {
        return true;
    }
    return false;
}

/**
 * @brief Returns whether classifier c1 is more general than c2.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier tested to be more general.
 * @param [in] c2 The classifier tested to be more specific.
 * @return Whether classifier c1 is more general than c2.
 */
bool
cl_general(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2)
{
    if (cond_general(xcsf, c1, c2)) {
        return act_general(xcsf, c1, c2);
    }
    return false;
}

/**
 * @brief Performs classifier mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier being mutated.
 * @return Whether any alterations were made.
 */
bool
cl_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    const bool cm = cond_mutate(xcsf, c);
    const bool pm = pred_mutate(xcsf, c);
    const bool am = (xcsf->n_actions > 1) ? act_mutate(xcsf, c) : false;
    if (cm || pm || am) {
        return true;
    }
    return false;
}

/**
 * @brief Performs classifier crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier being crossed.
 * @param [in] c2 The second classifier being crossed.
 * @return Whether any alterations were made.
 */
bool
cl_crossover(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2)
{
    const bool cc = cond_crossover(xcsf, c1, c2);
    const bool pc = pred_crossover(xcsf, c1, c2);
    const bool ac = act_crossover(xcsf, c1, c2);
    if (cc || pc || ac) {
        return true;
    }
    return false;
}

/**
 * @brief Returns the size of a classifier's condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition size to return.
 * @return The size of the condition.
 */
double
cl_cond_size(const struct XCSF *xcsf, const struct Cl *c)
{
    return cond_size(xcsf, c);
}

/**
 * @brief Returns the size of a classifier's prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction size to return.
 * @return The size of the prediction.
 */
double
cl_pred_size(const struct XCSF *xcsf, const struct Cl *c)
{
    return pred_size(xcsf, c);
}

/**
 * @brief Writes a classifier to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cl_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&c->err, sizeof(double), 1, fp);
    s += fwrite(&c->fit, sizeof(double), 1, fp);
    s += fwrite(&c->num, sizeof(int), 1, fp);
    s += fwrite(&c->exp, sizeof(int), 1, fp);
    s += fwrite(&c->size, sizeof(double), 1, fp);
    s += fwrite(&c->time, sizeof(int), 1, fp);
    s += fwrite(&c->m, sizeof(bool), 1, fp);
    s += fwrite(&c->age, sizeof(int), 1, fp);
    s += fwrite(&c->mtotal, sizeof(int), 1, fp);
    s += fwrite(c->prediction, sizeof(double), xcsf->y_dim, fp);
    s += fwrite(&c->action, sizeof(int), 1, fp);
    s += act_save(xcsf, c, fp);
    s += pred_save(xcsf, c, fp);
    s += cond_save(xcsf, c, fp);
    return s;
}

/**
 * @brief Reads a classifier from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cl_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    size_t s = 0;
    s += fread(&c->err, sizeof(double), 1, fp);
    s += fread(&c->fit, sizeof(double), 1, fp);
    s += fread(&c->num, sizeof(int), 1, fp);
    s += fread(&c->exp, sizeof(int), 1, fp);
    s += fread(&c->size, sizeof(double), 1, fp);
    s += fread(&c->time, sizeof(int), 1, fp);
    s += fread(&c->m, sizeof(bool), 1, fp);
    s += fread(&c->age, sizeof(int), 1, fp);
    s += fread(&c->mtotal, sizeof(int), 1, fp);
    c->prediction = malloc(sizeof(double) * xcsf->y_dim);
    s += fread(c->prediction, sizeof(double), xcsf->y_dim, fp);
    s += fread(&c->action, sizeof(int), 1, fp);
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c);
    s += act_load(xcsf, c, fp);
    s += pred_load(xcsf, c, fp);
    s += cond_load(xcsf, c, fp);
    return s;
}

/**
 * @brief Returns a json formatted string representation of a classifier.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier to be returned.
 * @param [in] return_cond Whether to return the condition.
 * @param [in] return_act Whether to return the action.
 * @param [in] return_pred Whether to return the prediction.
 * @return String encoded in json format.
 */
const char *
cl_json_export(const struct XCSF *xcsf, const struct Cl *c,
               const bool return_cond, const bool return_act,
               const bool return_pred)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "error", c->err);
    cJSON_AddNumberToObject(json, "fitness", c->fit);
    cJSON_AddNumberToObject(json, "accuracy", cl_acc(xcsf, c));
    cJSON_AddNumberToObject(json, "set_size", c->size);
    cJSON_AddNumberToObject(json, "numerosity", c->num);
    cJSON_AddNumberToObject(json, "experience", c->exp);
    cJSON_AddNumberToObject(json, "time", c->time);
    cJSON_AddNumberToObject(json, "samples_seen", c->age);
    cJSON_AddNumberToObject(json, "samples_matched", c->mtotal);
    cJSON_AddBoolToObject(json, "current_match", c->m);
    cJSON_AddNumberToObject(json, "current_action", c->action);
    cJSON *p = cJSON_CreateDoubleArray(c->prediction, xcsf->y_dim);
    cJSON_AddItemToObject(json, "current_prediction", p);
    if (return_cond) {
        cJSON *condition = cJSON_Parse(cond_json_export(xcsf, c));
        cJSON_AddItemToObject(json, "condition", condition);
    }
    if (return_act) {
        cJSON *action = cJSON_Parse(act_json_export(xcsf, c));
        cJSON_AddItemToObject(json, "action", action);
    }
    if (return_pred) {
        cJSON *prediction = cJSON_Parse(pred_json_export(xcsf, c));
        cJSON_AddItemToObject(json, "prediction", prediction);
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
