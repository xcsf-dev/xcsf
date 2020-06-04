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
 * @brief System-level functions for initialising, saving, loading, etc.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include "xcsf.h"
#include "loss.h"
#include "clset.h"
#include "cl.h"
#include "neural.h"
#include "prediction.h"
#include "condition.h"
#include "cond_neural.h"
#include "pred_neural.h"

static const double VERSION = 1.06; //!< XCSF version number

static size_t xcsf_load_params(XCSF *xcsf, FILE *fp);
static size_t xcsf_save_params(const XCSF *xcsf, FILE *fp);
static void xcsf_store_pop(XCSF *xcsf);

/**
 * @brief Initialises XCSF with an empty population.
 * @param xcsf The XCSF data structure.
 */
void xcsf_init(XCSF *xcsf)
{
    xcsf->time = 0;
    xcsf->msetsize = 0;
    xcsf->mfrac = 0;
    clset_init(&xcsf->pset);
    clset_init(&xcsf->prev_pset);
}

/**
 * @brief Prints the current XCSF population.
 * @param xcsf The XCSF data structure.
 * @param printc Whether to print condition structures.
 * @param printa Whether to print action structures.
 * @param printp Whether to print prediction structures.
 */
void xcsf_print_pop(const XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp)
{
    clset_print(xcsf, &xcsf->pset, printc, printa, printp);
}

/**
 * @brief Writes the current state of XCSF to a binary file.
 * @param xcsf The XCSF data structure.
 * @param fname The name of the output file.
 * @return The total number of elements written.
 */
size_t xcsf_save(const XCSF *xcsf, const char *fname)
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
size_t xcsf_load(XCSF *xcsf, const char *fname)
{
    if(xcsf->pset.size > 0) {
        clset_kill(xcsf, &xcsf->pset);
        clset_init(&xcsf->pset);
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
static size_t xcsf_save_params(const XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->time, sizeof(int), 1, fp);
    s += fwrite(&xcsf->msetsize, sizeof(double), 1, fp);
    s += fwrite(&xcsf->mfrac, sizeof(double), 1, fp);
    s += fwrite(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PERF_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fwrite(&xcsf->AUTO_ENCODE, sizeof(_Bool), 1, fp);
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
    s += fwrite(&xcsf->M_PROBATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->LAMBDA, sizeof(int), 1, fp);
    s += fwrite(&xcsf->EA_SELECT_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->EA_SELECT_SIZE, sizeof(double), 1, fp);
    s += fwrite(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_MAX, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_MIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_SMIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_BITS, sizeof(int), 1, fp);
    s += fwrite(xcsf->COND_NUM_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fwrite(xcsf->COND_MAX_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fwrite(&xcsf->COND_OUTPUT_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_HIDDEN_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(xcsf->PRED_NUM_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fwrite(xcsf->PRED_MAX_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fwrite(&xcsf->PRED_OUTPUT_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_HIDDEN_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_MOMENTUM, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->COND_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_EVOLVE_ETA, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->PRED_SGD_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->MAX_NEURON_MOD, sizeof(int), 1, fp);
    s += fwrite(&xcsf->DGP_NUM_NODES, sizeof(int), 1, fp);
    s += fwrite(&xcsf->RESET_STATES, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->MAX_K, sizeof(int), 1, fp);
    s += fwrite(&xcsf->MAX_T, sizeof(int), 1, fp);
    s += fwrite(&xcsf->COND_ETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_ETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_X0, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_RLS_SCALE_FACTOR, sizeof(double), 1, fp);
    s += fwrite(&xcsf->PRED_RLS_LAMBDA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->EA_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->SET_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->THETA_SUB, sizeof(int), 1, fp);
    s += fwrite(&xcsf->train, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->x_dim, sizeof(int), 1, fp);
    s += fwrite(&xcsf->y_dim, sizeof(int), 1, fp);
    s += fwrite(&xcsf->n_actions, sizeof(int), 1, fp);
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
    s += fread(&xcsf->mfrac, sizeof(double), 1, fp);
    s += fread(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->PERF_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fread(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fread(&xcsf->AUTO_ENCODE, sizeof(_Bool), 1, fp);
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
    s += fread(&xcsf->M_PROBATION, sizeof(int), 1, fp);
    s += fread(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fread(&xcsf->LAMBDA, sizeof(int), 1, fp);
    s += fread(&xcsf->EA_SELECT_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->EA_SELECT_SIZE, sizeof(double), 1, fp);
    s += fread(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_MAX, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_MIN, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_SMIN, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_BITS, sizeof(int), 1, fp);
    s += fread(xcsf->COND_NUM_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fread(xcsf->COND_MAX_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fread(&xcsf->COND_OUTPUT_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->COND_HIDDEN_ACTIVATION, sizeof(int), 1, fp);
    s += fread(xcsf->PRED_NUM_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fread(xcsf->PRED_MAX_NEURONS, sizeof(int), MAX_LAYERS, fp);
    s += fread(&xcsf->PRED_OUTPUT_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_HIDDEN_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_MOMENTUM, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->COND_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_NEURONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_FUNCTIONS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_EVOLVE_ETA, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->PRED_SGD_WEIGHTS, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->MAX_NEURON_MOD, sizeof(int), 1, fp);
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
    s += fread(&xcsf->x_dim, sizeof(int), 1, fp);
    s += fread(&xcsf->y_dim, sizeof(int), 1, fp);
    s += fread(&xcsf->n_actions, sizeof(int), 1, fp);
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

/**
 * @brief Expands the autoencoders in the population.
 * @param xcsf The XCSF data structure.
 */
void xcsf_ae_expand(XCSF *xcsf)
{
    xcsf_store_pop(xcsf);
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        pred_neural_ae_expand(xcsf, iter->cl);
        iter->cl->fit = xcsf->INIT_FITNESS;
        iter->cl->err = xcsf->INIT_ERROR;
        iter->cl->exp = 0;
        iter->cl->time = xcsf->time;
    }
}

/**
 * @brief Switches from autoencoding to classification.
 * @param xcsf The XCSF data structure.
 * @param y_dim The output dimension (i.e., the number of classes).
 */
void xcsf_ae_to_classifier(XCSF *xcsf, int y_dim)
{
    xcsf_store_pop(xcsf);
    xcsf->AUTO_ENCODE = false;
    xcsf->y_dim = y_dim;
    xcsf->LOSS_FUNC = 5; // one-hot encoding error
    loss_set_func(xcsf);
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        free(iter->cl->prediction);
        iter->cl->prediction = calloc(xcsf->y_dim, sizeof(double));
        pred_neural_ae_to_classifier(xcsf, iter->cl);
        iter->cl->fit = xcsf->INIT_FITNESS;
        iter->cl->err = xcsf->INIT_ERROR;
        iter->cl->exp = 0;
        iter->cl->time = xcsf->time;
    }
}

/**
 * @brief Stores the current population.
 * @param xcsf The XCSF data structure.
 */
static void xcsf_store_pop(XCSF *xcsf)
{
    if(xcsf->prev_pset.size > 0) {
        clset_kill(xcsf, &xcsf->prev_pset);
        clset_init(&xcsf->prev_pset);
    }
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        CL *new = malloc(sizeof(CL));
        CL *src = iter->cl;
        cl_init_copy(xcsf, new, src);
        clset_add(&xcsf->prev_pset, new);
    }
}
