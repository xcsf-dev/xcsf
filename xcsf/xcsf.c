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
 * @file xcsf.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief High level XCSF functions for executing training, predicting, saving
 * and reloading the system from persistent storage, etc.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include "xcsf.h"
#include "utils.h"
#include "loss.h"
#include "perf.h"
#include "cl.h"
#include "clset.h"
#include "ea.h"

static const double VERSION = 1.06; //!< XCSF version number

static double xcsf_learn_trial(XCSF *xcsf, double *pred, double *x, double *y);
static double xcsf_test_trial(XCSF *xcsf, double *pred, double *x, double *y);
static size_t xcsf_load_params(XCSF *xcsf, FILE *fp);
static size_t xcsf_save_params(XCSF *xcsf, FILE *fp);

/**
 * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
 * training data.
 * @param xcsf The XCSF data structure.
 * @param train_data The input data to use for training.
 * @param shuffle Whether to randomise the instances during training.
 * @return The average XCSF training error using the loss function.
 */
double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle)
{  
    INPUT *test_data = NULL;
    return xcsf_fit2(xcsf, train_data, test_data, shuffle);
}

/**
 * @brief Executes MAX_TRIALS number of XCSF learning iterations using the training.
 * data and test iterations using the test data.
 * @param xcsf The XCSF data structure.
 * @param train_data The input data to use for training.
 * @param test_data The input data to use for testing.
 * @param shuffle Whether to randomise the instances during training.
 * @return The average XCSF training error using the loss function.
 */
double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle)
{   
    gplot_init(xcsf);
    double perr = 0;
    double err = 0;
    double pterr = 0;
    double *pred = malloc(sizeof(double) * xcsf->num_y_vars);
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        // select training sample
        int row = 0;
        if(shuffle) {
            row = irand_uniform(0, train_data->rows);
        }
        else {
            row = (cnt % train_data->rows + train_data->rows) % train_data->rows;
        }     	
        double *x = &train_data->x[row * train_data->x_cols];
        double *y = &train_data->y[row * train_data->y_cols];
        // execute a learning trial
        xcsf->train = true;
        double error = xcsf_learn_trial(xcsf, pred, x, y);
        perr += error; 
        err += error;
        // if some test data has been supplied
        if(test_data != NULL) {
            // select test sample
            if(shuffle) {
                row = irand_uniform(0, test_data->rows);
            }
            else {
                row = (cnt % test_data->rows + test_data->rows) % test_data->rows;
            }
            x = &test_data->x[row * test_data->x_cols];
            y = &test_data->y[row * test_data->y_cols];
            // execute a test trial
            xcsf->train = false;
            pterr += xcsf_test_trial(xcsf, pred, x, y);
            // display performance as necessary (training and testing)
            if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
                disp_perf2(xcsf, perr/xcsf->PERF_AVG_TRIALS, pterr/xcsf->PERF_AVG_TRIALS, cnt);
                perr = 0; pterr = 0;
            }
        }
        // display performance as necessary (training only)
        else if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf1(xcsf, perr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0;
        }
    }
    // clean up
    free(pred);
    gplot_free(xcsf);
    return err/xcsf->MAX_TRIALS;
}

/**
 * @brief Executes a single XCSF learning trial.
 * @param xcsf The XCSF data structure.
 * @param pred The calculated XCSF prediction (set by this function).
 * @param x The feature variables.
 * @param y The labelled variables.
 * @return The XCSF training error using the loss function.
 */
static double xcsf_learn_trial(XCSF *xcsf, double *pred, double *x, double *y)
{
    SET mset; // match set
    SET kset; // kill set
    clset_init(xcsf, &mset);
    clset_init(xcsf, &kset);
    clset_match(xcsf, &mset, &kset, x);
    clset_pred(xcsf, &mset, x, pred);
    clset_update(xcsf, &mset, &kset, x, y, true);
    ea(xcsf, &mset, &kset);
    xcsf->time += 1;
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    clset_kill(xcsf, &kset); // kills deleted classifiers
    clset_free(xcsf, &mset); // frees the match set list
    return (xcsf->loss_ptr)(xcsf, pred, y);
}

/**
 * @brief Executes a single XCSF testing trial.
 * @param xcsf The XCSF data structure.
 * @param pred The calculated XCSF prediction (set by this function).
 * @param x The feature variables.
 * @param y The labelled variables.
 * @return The XCSF testing error using the loss function.
 */
static double xcsf_test_trial(XCSF *xcsf, double *pred, double *x, double *y)
{
    SET mset; // match set
    SET kset; // kill set
    clset_init(xcsf, &mset);
    clset_init(xcsf, &kset);
    clset_match(xcsf, &mset, &kset, x);
    clset_pred(xcsf, &mset, x, pred);
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    clset_kill(xcsf, &kset); // kills deleted classifiers
    clset_free(xcsf, &mset); // frees the match set list  
    return (xcsf->loss_ptr)(xcsf, pred, y);
}

/**
 * @brief Calculates the XCSF predictions for the provided input.
 * @param xcsf The XCSF data structure.
 * @param input The input feature variables.
 * @param output The calculated XCSF predictions (set by this function).
 * @param rows The number of instances.
 */
void xcsf_predict(XCSF *xcsf, double *input, double *output, int rows)
{   
    xcsf->train = false;
    for(int row = 0; row < rows; row++) {
        SET mset; // match set
        SET kset; // kill set
        clset_init(xcsf, &mset);
        clset_init(xcsf, &kset);
        clset_match(xcsf, &mset, &kset, &input[row * xcsf->num_x_vars]);
        clset_pred(xcsf, &mset, &input[row * xcsf->num_x_vars], &output[row * xcsf->num_y_vars]);
        clset_kill(xcsf, &kset); // kills deleted classifiers
        clset_free(xcsf, &mset); // frees the match set list
    }
}

/**
 * @brief Calculates the XCSF error for the input data.
 * @param xcsf The XCSF data structure.
 * @param test_data The input data to calculate the error.
 * @return The average XCSF error using the loss function.
 */
double xcsf_score(XCSF *xcsf, INPUT *test_data)
{
    xcsf->train = false;
    double err = 0;
    double *pred = malloc(sizeof(double) * xcsf->num_y_vars);
    for(int row = 0; row < test_data->rows; row++) {
        double *x = &test_data->x[row * test_data->x_cols];
        double *y = &test_data->y[row * test_data->y_cols];
        err += xcsf_test_trial(xcsf, pred, x, y);
    }
    free(pred);
    return err/(double)test_data->rows;
}

/**
 * @brief Prints the current XCSF population.
 * @param xcsf The XCSF data structure.
 * @param printc Whether to print condition structures.
 * @param printa Whether to print action structures.
 * @param printp Whether to print prediction structures.
 */
void xcsf_print_pop(XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp)
{
    clset_print(xcsf, &xcsf->pset, printc, printa, printp);
}

/**
 * @brief Prints the XCSF match set for the supplied input.
 * @param xcsf The XCSF data structure.
 * @param input The input features to perform matching.
 * @param printc Whether to print condition structures.
 * @param printa Whether to print action structures.
 * @param printp Whether to print prediction structures.
 */
void xcsf_print_match_set(XCSF *xcsf, double *input, _Bool printc, _Bool printa, _Bool printp)
{
    SET mset; // match set
    SET kset; // kill set
    clset_init(xcsf, &mset);
    clset_init(xcsf, &kset);
    clset_match(xcsf, &mset, &kset, input);
    clset_print(xcsf, &mset, printc, printa, printp);
    clset_kill(xcsf, &kset); // kills deleted classifiers
    clset_free(xcsf, &mset); // frees the match set list
}

/**
 * @brief Writes the current state of XCSF to a binary file.
 * @param xcsf The XCSF data structure.
 * @param fname The name of the output file.
 * @return The total number of elements written.
 */
size_t xcsf_save(XCSF *xcsf, char *fname)
{
    FILE *fp = fopen(fname, "wb");
    if(fp == 0) {
        printf("Error opening save file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0;
    s += fwrite(&VERSION, sizeof(double), 1, fp);
    s += xcsf_save_params(xcsf, fp);
    s += clset_pop_save(xcsf, fp);
    fclose(fp);
    return s;
}

/**
 * @brief Reads the state of XCSF from a binary file.
 * @param xcsf The XCSF data structure.
 * @param fname The name of the input file.
 * @return The total number of elements read.
 */
size_t xcsf_load(XCSF *xcsf, char *fname)
{
    if(xcsf->pset.size > 0) {
        clset_kill(xcsf, &xcsf->pset);
        clset_init(xcsf, &xcsf->pset);
    }
    FILE *fp = fopen(fname, "rb");
    if(fp == 0) {
        printf("Error opening load file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0;
    double version = 0;
    s += fread(&version, sizeof(double), 1, fp);
    if(version != VERSION) {
        printf("Error loading file: %s. Version mismatch. ", fname);
        printf("This version: %f.\nLoaded version: %f", VERSION, version);
        exit(EXIT_FAILURE);
    }
    s += xcsf_load_params(xcsf, fp);
    s += clset_pop_load(xcsf, fp);
    fclose(fp);
    return s;
}

/**
 * @brief Writes the XCSF data structure to a binary file.
 * @param xcsf The XCSF data structure.
 * @param fp Pointer to the output file.
 * @return The total number of elements written.
 */
static size_t xcsf_save_params(XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->time, sizeof(int), 1, fp);
    s += fwrite(&xcsf->msetsize, sizeof(double), 1, fp);
    s += fwrite(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PERF_AVG_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ALPHA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->BETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->DELTA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->EPS_0, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ERR_REDUC, sizeof(double), 1, fp);
    s += fwrite(&xcsf->FIT_REDUC, sizeof(double), 1, fp);
    s += fwrite(&xcsf->INIT_ERROR, sizeof(double), 1, fp);
    s += fwrite(&xcsf->INIT_FITNESS, sizeof(double), 1, fp);
    s += fwrite(&xcsf->NU, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_DEL, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ACT_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fwrite(&xcsf->P_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->F_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->S_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->E_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->LAMBDA, sizeof(int), 1, fp);
    s += fwrite(&xcsf->EA_SELECT_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->EA_SELECT_SIZE, sizeof(double), 1, fp);
    s += fwrite(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->SAM_NUM, sizeof(int), 1, fp);
    s += fwrite(&xcsf->SAM_MIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_MAX, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_MIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_SMIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_BITS, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_MAX_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_MAX_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_MOMENTUM, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_ETA, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_SGD_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->DGP_NUM_NODES, sizeof(int), 1, fp);
    s += fwrite(&xcsf->RESET_STATES, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->MAX_K, sizeof(int), 1, fp);
    s += fwrite(&xcsf->MAX_T, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_X0, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_ETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_ETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_RLS_SCALE_FACTOR, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_RLS_LAMBDA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->EA_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->SET_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->THETA_SUB, sizeof(int), 1, fp);
    s += fwrite(&xcsf->train, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->num_x_vars, sizeof(int), 1, fp);
    s += fwrite(&xcsf->num_y_vars, sizeof(int), 1, fp);
    s += fwrite(&xcsf->num_actions, sizeof(int), 1, fp);
    s += fwrite(&xcsf->GP_NUM_CONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->GP_INIT_DEPTH, sizeof(int), 1, fp);
    s += fwrite(xcsf->gp_cons, sizeof(double), xcsf->GP_NUM_CONS, fp);
    return s;
}

/**
 * @brief Reads the XCSF data structure from a binary file.
 * @param xcsf The XCSF data structure.
 * @param fp Pointer to the input file.
 * @return The total number of elements read.
 */
static size_t xcsf_load_params(XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fread(&xcsf->time, sizeof(int), 1, fp);
    s += fread(&xcsf->msetsize, sizeof(double), 1, fp);
    s += fread(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->PERF_AVG_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fread(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fread(&xcsf->ALPHA, sizeof(double), 1, fp);
    s += fread(&xcsf->BETA, sizeof(double), 1, fp);
    s += fread(&xcsf->DELTA, sizeof(double), 1, fp);
    s += fread(&xcsf->EPS_0, sizeof(double), 1, fp);
    s += fread(&xcsf->ERR_REDUC, sizeof(double), 1, fp);
    s += fread(&xcsf->FIT_REDUC, sizeof(double), 1, fp);
    s += fread(&xcsf->INIT_ERROR, sizeof(double), 1, fp);
    s += fread(&xcsf->INIT_FITNESS, sizeof(double), 1, fp);
    s += fread(&xcsf->NU, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_DEL, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->ACT_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fread(&xcsf->P_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->F_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->S_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->E_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fread(&xcsf->LAMBDA, sizeof(int), 1, fp);
    s += fread(&xcsf->EA_SELECT_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->EA_SELECT_SIZE, sizeof(double), 1, fp);
    s += fread(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->SAM_NUM, sizeof(int), 1, fp);
    s += fread(&xcsf->SAM_MIN, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_MAX, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_MIN, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_SMIN, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_BITS, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_MAX_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_MAX_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_MOMENTUM, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_ETA, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_SGD_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->DGP_NUM_NODES, sizeof(int), 1, fp);
    s += fread(&xcsf->RESET_STATES, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->MAX_K, sizeof(int), 1, fp);
    s += fread(&xcsf->MAX_T, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_ETA, sizeof(double), 1, fp);
    s += fread(&xcsf->PRED_ETA, sizeof(double), 1, fp);
    s += fread(&xcsf->PRED_X0, sizeof(double), 1, fp);
    s += fread(&xcsf->PRED_RLS_SCALE_FACTOR, sizeof(double), 1, fp);
    s += fread(&xcsf->PRED_RLS_LAMBDA, sizeof(double), 1, fp);
    s += fread(&xcsf->EA_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->SET_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->THETA_SUB, sizeof(int), 1, fp);
    s += fread(&xcsf->train, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->num_x_vars, sizeof(int), 1, fp);
    s += fread(&xcsf->num_y_vars, sizeof(int), 1, fp);
    s += fread(&xcsf->num_actions, sizeof(int), 1, fp);
    s += fread(&xcsf->GP_NUM_CONS, sizeof(int), 1, fp);
    s += fread(&xcsf->GP_INIT_DEPTH, sizeof(int), 1, fp);
    free(xcsf->gp_cons); // always malloced on start
    xcsf->gp_cons = malloc(sizeof(double)*xcsf->GP_NUM_CONS);
    s += fread(xcsf->gp_cons, sizeof(double), xcsf->GP_NUM_CONS, fp);
    loss_set_func(xcsf);         
    return s;
}

/**
 * @brief Returns the XCSF version number.
 * @return version number.
 */  
double xcsf_version()
{
    return VERSION;
}
