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
 * @date 2015--2019.
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
#include "sam.h"
#include "condition.h"
#include "prediction.h"
#include "action.h"
#include "cl.h"

double cl_update_err(XCSF *xcsf, CL *c, double *y, _Bool current);
double cl_update_size(XCSF *xcsf, CL *c, double num_sum);

/**
 * @brief Initialises a new classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier data structure to initialise.
 * @param size The initial set size value.
 * @param time The current number of XCSF learning trials.
 */
void cl_init(XCSF *xcsf, CL *c, int size, int time)
{
    c->fit = xcsf->INIT_FITNESS;
    c->err = xcsf->INIT_ERROR;
    c->num = 1;
    c->exp = 0;
    c->size = size;
    c->time = time;
    c->prediction = calloc(xcsf->num_y_vars, sizeof(double));
    c->prev_prediction = calloc(xcsf->num_y_vars, sizeof(double));
    c->action = 0;
    c->m = false;
    c->mhist = calloc(xcsf->THETA_SUB, sizeof(_Bool));
    sam_init(xcsf, &c->mu);
}

/**
 * @brief Copies the condition, action, and prediction from
 * one classifier to another.
 * @param xcsf The XCSF data structure.
 * @param to The destination classifier.
 * @param from The source classifier.
 */
void cl_copy(XCSF *xcsf, CL *to, CL *from)
{
    to->cond_vptr = from->cond_vptr;
    to->pred_vptr = from->pred_vptr;
    to->act_vptr = from->act_vptr;
    sam_copy(xcsf, to->mu, from->mu);
    act_copy(xcsf, to, from);
    cond_copy(xcsf, to, from);
    pred_copy(xcsf, to, from);
}

/**
 * @brief Covers the condition and action for a classifier.
 * @param xcsf The XCSF data structure.
 * @param c The classifier being covered.
 * @param x The input state to cover.
 * @param action The action to cover.
 */
void cl_cover(XCSF *xcsf, CL *c, double *x, int action)
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
void cl_rand(XCSF *xcsf, CL *c)
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
double cl_del_vote(XCSF *xcsf, CL *c, double avg_fit)
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
double cl_acc(XCSF *xcsf, CL *c)
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
 * @param current Whether the payoff is for the current or previous state.
 * @param set_num The number of micro-classifiers in the set.
 */
void cl_update(XCSF *xcsf, CL *c, double *x, double *y, int set_num, _Bool current)
{
    c->exp++;
    cl_update_err(xcsf, c, y, current);
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
 * @param current Whether the payoff is for the current or previous state.
 * @return Error multiplied by numerosity.
 */
double cl_update_err(XCSF *xcsf, CL *c, double *y, _Bool current)
{
    double error = 0;
    if(current) {
        error = (xcsf->loss_ptr)(xcsf, c->prediction, y);
    }
    else {
        error = (xcsf->loss_ptr)(xcsf, c->prev_prediction, y);
    }
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
void cl_update_fit(XCSF *xcsf, CL *c, double acc_sum, double acc)
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
double cl_update_size(XCSF *xcsf, CL *c, double num_sum)
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
void cl_free(XCSF *xcsf, CL *c)
{
    free(c->mhist);
    free(c->prediction);
    free(c->prev_prediction);
    sam_free(xcsf, c->mu);
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
void cl_print(XCSF *xcsf, CL *c, _Bool printc, _Bool printa, _Bool printp)
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
    printf("err=%f, fit=%f, num=%d, exp=%d, size=%f, time=%d\n", 
            c->err, c->fit, c->num, c->exp, c->size, c->time);
}  

/**
 * @brief Calculates whether the classifier matches the input. 
 * @param xcsf The XCSF data structure.
 * @param c The classifier to match.
 * @param x The input state.
 * @return Whether the classifier matches the input.
 */
_Bool cl_match(XCSF *xcsf, CL *c, double *x)
{
    _Bool m = cond_match(xcsf, c, x);
    c->mhist[c->exp % xcsf->THETA_SUB] = m;
    return m;
}

/**
 * @brief Returns whether the classifier matched the most recent input. 
 * @param xcsf The XCSF data structure.
 * @param c The classifier to match.
 * @return Whether the classifier matched the most recent input.
 */  
_Bool cl_m(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    return c->m;
}

/**
 * @brief Computes the current classifier payoff prediction using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier making the prediction.
 * @param x The input state.
 * @return The classifier's payoff predictions.
 */
double *cl_predict(XCSF *xcsf, CL *c, double *x)
{
    memcpy(c->prev_prediction, c->prediction, sizeof(double) * xcsf->num_y_vars);
    return pred_compute(xcsf, c, x);
}

/**
 * @brief Returns whether the classifier is a potential subsumer.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to print.
 * @return Whether the classifier is an eligible subsumer.
 */
_Bool cl_subsumer(XCSF *xcsf, CL *c)
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
_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2)
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
_Bool cl_mutate(XCSF *xcsf, CL *c)
{
    if(xcsf->SAM_NUM > 0) {
        xcsf->S_MUTATION = c->mu[0];
        if(xcsf->SAM_NUM > 1) {
            xcsf->P_MUTATION = c->mu[1];
            if(xcsf->SAM_NUM > 2) {
                xcsf->E_MUTATION = c->mu[2];
                if(xcsf->SAM_NUM > 3) {
                    xcsf->F_MUTATION = c->mu[3];
                }
            }
        }
    } 
    _Bool cm = cond_mutate(xcsf, c);
    _Bool pm = pred_mutate(xcsf, c);
    _Bool am = act_mutate(xcsf, c);
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
_Bool cl_crossover(XCSF *xcsf, CL *c1, CL *c2)
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
double cl_mutation_rate(XCSF *xcsf, CL *c, int m)
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
int cl_cond_size(XCSF *xcsf, CL *c)
{
    return cond_size(xcsf, c);
}
                     
/**
 * @brief Returns the size of the classifier prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction size to return.
 * @return The size of the prediction.
 */
int cl_pred_size(XCSF *xcsf, CL *c)
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
size_t cl_save(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    s += fwrite(c->mu, sizeof(double), xcsf->SAM_NUM, fp);
    s += fwrite(&c->err, sizeof(double), 1, fp);
    s += fwrite(&c->fit, sizeof(double), 1, fp);
    s += fwrite(&c->num, sizeof(int), 1, fp);
    s += fwrite(&c->exp, sizeof(int), 1, fp);
    s += fwrite(&c->size, sizeof(double), 1, fp);
    s += fwrite(&c->time, sizeof(int), 1, fp);
    s += fwrite(&c->m, sizeof(_Bool), 1, fp);
    s += fwrite(c->mhist, sizeof(_Bool), xcsf->THETA_SUB, fp);
    s += fwrite(c->prediction, sizeof(double), xcsf->num_y_vars, fp);
    s += fwrite(c->prev_prediction, sizeof(double), xcsf->num_y_vars, fp);
    s += fwrite(&c->action, sizeof(int), 1, fp);
    s += act_save(xcsf, c, fp);
    s += pred_save(xcsf, c, fp);
    s += cond_save(xcsf, c, fp);
    //printf("cl saved %lu elements\n", (unsigned long)s);
    return s;
}

/**
 * @brief Reads the classifier from a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to load.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t cl_load(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    c->mu = malloc(xcsf->SAM_NUM * sizeof(double));
    s += fread(c->mu, sizeof(double), xcsf->SAM_NUM, fp);
    s += fread(&c->err, sizeof(double), 1, fp);
    s += fread(&c->fit, sizeof(double), 1, fp);
    s += fread(&c->num, sizeof(int), 1, fp);
    s += fread(&c->exp, sizeof(int), 1, fp);
    s += fread(&c->size, sizeof(double), 1, fp);
    s += fread(&c->time, sizeof(int), 1, fp);
    s += fread(&c->m, sizeof(_Bool), 1, fp);
    c->mhist = malloc(xcsf->THETA_SUB * sizeof(_Bool));
    s += fread(c->mhist, sizeof(_Bool), xcsf->THETA_SUB, fp);
    c->prediction = malloc(xcsf->num_y_vars * sizeof(double));
    s += fread(c->prediction, sizeof(double), xcsf->num_y_vars, fp);
    c->prev_prediction = malloc(xcsf->num_y_vars * sizeof(double));
    s += fread(c->prev_prediction, sizeof(double), xcsf->num_y_vars, fp);
    s += fread(&c->action, sizeof(int), 1, fp);
    action_set(xcsf, c);
    prediction_set(xcsf, c);
    condition_set(xcsf, c);
    s += act_load(xcsf, c, fp);
    s += pred_load(xcsf, c, fp);
    s += cond_load(xcsf, c, fp);
    //printf("cl loaded %lu elements\n", (unsigned long)s);
    return s;
}
