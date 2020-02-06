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
 * @date 2015--2020.
 * @brief Functions operating on classifiers.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "loss.h"
#include "condition.h"
#include "prediction.h"
#include "action.h"
#include "cl.h"

static double cl_update_err(const XCSF *xcsf, CL *c, const double *y);
static double cl_update_size(const XCSF *xcsf, CL *c, double num_sum);

/**
 * @brief Initialises a new classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier data structure to initialise.
 * @param size The initial set size value.
 * @param time The current number of XCSF learning trials.
 */
void cl_init(const XCSF *xcsf, CL *c, int size, int time)
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
 * @brief Copies the condition, action, and prediction from
 * one classifier to another.
 * @param xcsf The XCSF data structure.
 * @param to The destination classifier.
 * @param from The source classifier.
 */
void cl_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    to->cond_vptr = from->cond_vptr;
    to->pred_vptr = from->pred_vptr;
    to->act_vptr = from->act_vptr;
    act_copy(xcsf, to, from);
    cond_copy(xcsf, to, from);
    if(xcsf->PRED_RESET) {
        pred_init(xcsf, to);
    }
    else {
        pred_copy(xcsf, to, from);
    }
}

/**
 * @brief Covers the condition and action for a classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier being covered.
 * @param x The input state to cover.
 * @param action The action to cover.
 */
void cl_cover(const XCSF *xcsf, CL *c, const double *x, int action)
{
    cl_rand(xcsf, c);
    cond_cover(xcsf, c, x);
    act_cover(xcsf, c, x, action);
}

/**
 * @brief Initialises random actions, conditions and predictions.
 * @param xcsf The XCSF data structure.
 * @param c The classifier being randomly initialised.
 */
void cl_rand(const XCSF *xcsf, CL *c)
{
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c); 
    cond_init(xcsf, c);
    pred_init(xcsf, c);
    act_init(xcsf, c);
}

/**
 * @brief Returns the deletion vote of the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to calculate the deletion vote.
 * @param avg_fit The population mean fitness.
 * @return The classifier's deletion vote. 
 */
double cl_del_vote(const XCSF *xcsf, const CL *c, double avg_fit)
{
    if(c->exp > xcsf->THETA_DEL && c->fit / c->num < xcsf->DELTA * avg_fit) {
        return c->size * c->num * avg_fit / (c->fit / c->num);
    }
    return c->size * c->num;
}

/**
 * @brief Returns the accuracy of the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier.
 * @return The classifier's accuracy. 
 */
double cl_acc(const XCSF *xcsf, const CL *c)
{
    if(c->err > xcsf->EPS_0) {
        return xcsf->ALPHA * pow(c->err / xcsf->EPS_0, -(xcsf->NU));
    }
    return 1;
}

/**
 * @brief Updates the classifier's parameters as well as condition, action, and
 * prediction depending on the knowledge representation.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to update.
 * @param x The input state.
 * @param y The payoff value.
 * @param set_num The number of micro-classifiers in the set.
 * @param cur Whether the payoff is for the current or previous state.
 */
void cl_update(const XCSF *xcsf, CL *c, const double *x, const double *y, int set_num, _Bool cur)
{
    c->exp++;
    // propagate inputs for the previous state update
    if(cur == false) {
        cl_predict(xcsf, c, x);
    }
    cl_update_err(xcsf, c, y);
    cl_update_size(xcsf, c, set_num);
    cond_update(xcsf, c, x, y);
    pred_update(xcsf, c, x, y);
    act_update(xcsf, c, x, y);
}

/**
 * @brief Updates the error of the classifier using the payoff.
 * @pre Classifier prediction must have been updated for the input state.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to update.
 * @param y The payoff value.
 * @return Error multiplied by numerosity.
 */
static double cl_update_err(const XCSF *xcsf, CL *c, const double *y)
{
    double error = (xcsf->loss_ptr)(xcsf, c->prediction, y);
    if(c->exp < 1 / xcsf->BETA) {
        c->err = (c->err * (c->exp - 1) + error) / c->exp;
    }
    else {
        c->err += xcsf->BETA * (error - c->err);
    }
    return c->err * c->num;
}

/**
 * @brief Updates the fitness of the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to update.
 * @param acc_sum The sum of all the accuracies in the set.
 * @param acc The accuracy of the classifier.
 */
void cl_update_fit(const XCSF *xcsf, CL *c, double acc_sum, double acc)
{
    c->fit += xcsf->BETA * ((acc * c->num) / acc_sum - c->fit);
}

/**
 * @brief Updates the set size estimate for the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to update.
 * @param num_sum The number of micro-classifiers in the set.
 * @return Set size multiplied by numerosity.
 */
static double cl_update_size(const XCSF *xcsf, CL *c, double num_sum)
{
    if(c->exp < 1 / xcsf->BETA) {
        c->size = (c->size * (c->exp - 1) + num_sum) / c->exp;
    }
    else {
        c->size += xcsf->BETA * (num_sum - c->size);
    }
    return c->size * c->num;
}

/**
 * @brief Frees the memory used by the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to free.
 */
void cl_free(const XCSF *xcsf, CL *c)
{
    free(c->prediction);
    cond_free(xcsf, c);
    act_free(xcsf, c);
    pred_free(xcsf, c);
    free(c);
}

/**
 * @brief Prints the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to print.
 * @param printc Whether to print the condition.
 * @param printa Whether to print the action.
 * @param printp Whether to print the prediction.
 */
void cl_print(const XCSF *xcsf, const CL *c, _Bool printc, _Bool printa, _Bool printp)
{
    if(printc || printa || printp) {
        printf("***********************************************\n");
        if(printc) {
            printf("\nCONDITION\n");
            cond_print(xcsf, c);
        }
        if(printp) {
            printf("\nPREDICTOR\n");
            pred_print(xcsf, c);
        }
        if(printa) {
            printf("\nACTION\n");
            act_print(xcsf, c);
        }
        printf("\n");
    }
    printf("err=%f fit=%f num=%d exp=%d size=%f time=%d age=%d mfrac=%f\n",
            c->err, c->fit, c->num, c->exp, c->size, c->time, c->age, cl_mfrac(xcsf, c));
}  

/**
 * @brief Calculates whether the classifier matches the input. 
 * @param xcsf The XCSF data structure.
 * @param c The classifier to match.
 * @param x The input state.
 * @return Whether the classifier matches the input.
 */
_Bool cl_match(const XCSF *xcsf, CL *c, const double *x)
{
    c->m = cond_match(xcsf, c, x);
    if(c->m) {
        c->mtotal++;
    }
    c->age++;
    return c->m;
}

/**
 * @brief Returns the fraction of observed inputs matched by the classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to match.
 * @return The fraction of matching inputs.
 */
double cl_mfrac(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    if(c->age > 0) {
        return (double) c->mtotal / c->age;
    }
    return 0;
}

/**
 * @brief Returns whether the classifier matched the most recent input. 
 * @param xcsf The XCSF data structure.
 * @param c The classifier to match.
 * @return Whether the classifier matched the most recent input.
 */  
_Bool cl_m(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    return c->m;
}
 
/**
 * @brief Computes the current classifier action using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier calculating the action.
 * @param x The input state.
 * @return The classifier's action.
 */
int cl_action(const XCSF *xcsf, CL *c, const double *x)
{
    c->action = act_compute(xcsf, c, x);
    return c->action;
}
 
/**
 * @brief Computes the current classifier payoff prediction using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier making the prediction.
 * @param x The input state.
 * @return The classifier's payoff predictions.
 */
const double *cl_predict(const XCSF *xcsf, const CL *c, const double *x)
{
    return pred_compute(xcsf, c, x);
}

/**
 * @brief Returns whether the classifier is a potential subsumer.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to print.
 * @return Whether the classifier is an eligible subsumer.
 */
_Bool cl_subsumer(const XCSF *xcsf, const CL *c)
{
    if(c->exp > xcsf->THETA_SUB && c->err < xcsf->EPS_0) {
        return true;
    }
    return false;
}

/**
 * @brief Returns whether classifier c1 is more general than c2.
 * @param xcsf The XCSF data structure.
 * @param c1 The classifier tested to be more general.
 * @param c2 The classifier tested to be more specific.
 * @return Whether c1 is more general than c2.
 */
_Bool cl_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    if(cond_general(xcsf, c1, c2)) {
        return act_general(xcsf, c1, c2);
    }
    return false;
} 

/**
 * @brief Performs classifier mutation.
 * @param xcsf The XCSF data structure.
 * @param c The classifier being mutated.
 * @return Whether any alterations were made.
 */
_Bool cl_mutate(const XCSF *xcsf, const CL *c)
{
    _Bool cm = cond_mutate(xcsf, c);
    _Bool pm = pred_mutate(xcsf, c);
    _Bool am = false;
    // skip action mutation for regression
    if(xcsf->n_actions > 1) {
        am = act_mutate(xcsf, c);
    }
    if(cm || pm || am) {
        return true;
    }
    return false;
}

/**
 * @brief Performs classifier crossover.
 * @param xcsf The XCSF data structure.
 * @param c1 The first classifier being crossed.
 * @param c2 The second classifier being crossed.
 * @return Whether any alterations were made.
 */
_Bool cl_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    _Bool cc = cond_crossover(xcsf, c1, c2);
    _Bool pc = pred_crossover(xcsf, c1, c2);
    _Bool ac = act_crossover(xcsf, c1, c2);
    if(cc || pc || ac) {
        return true;
    }
    return false;
}

/**
 * @brief Returns the classifier self-adaptive mutation rate.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose rate to return.
 * @param m Which mutation rate to return.
 * @return The current mutation rate.
 */
double cl_mutation_rate(const XCSF *xcsf, const CL *c, int m)
{
    (void)xcsf;
    return c->mu[m];
}  

/**
 * @brief Returns the size of the classifier condition.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose condition size to return.
 * @return The size of the condition.
 */
int cl_cond_size(const XCSF *xcsf, const CL *c)
{
    return cond_size(xcsf, c);
}
                     
/**
 * @brief Returns the size of the classifier prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction size to return.
 * @return The size of the prediction.
 */
int cl_pred_size(const XCSF *xcsf, const CL *c)
{
    return pred_size(xcsf, c);
}

/**
 * @brief Writes the classifier to a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to save.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t cl_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&c->err, sizeof(double), 1, fp);
    s += fwrite(&c->fit, sizeof(double), 1, fp);
    s += fwrite(&c->num, sizeof(int), 1, fp);
    s += fwrite(&c->exp, sizeof(int), 1, fp);
    s += fwrite(&c->size, sizeof(double), 1, fp);
    s += fwrite(&c->time, sizeof(int), 1, fp);
    s += fwrite(&c->m, sizeof(_Bool), 1, fp);
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
 * @brief Reads the classifier from a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to load.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t cl_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    s += fread(&c->err, sizeof(double), 1, fp);
    s += fread(&c->fit, sizeof(double), 1, fp);
    s += fread(&c->num, sizeof(int), 1, fp);
    s += fread(&c->exp, sizeof(int), 1, fp);
    s += fread(&c->size, sizeof(double), 1, fp);
    s += fread(&c->time, sizeof(int), 1, fp);
    s += fread(&c->m, sizeof(_Bool), 1, fp);
    s += fread(&c->age, sizeof(int), 1, fp);
    s += fread(&c->mtotal, sizeof(int), 1, fp);
    c->prediction = malloc(xcsf->y_dim * sizeof(double));
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
