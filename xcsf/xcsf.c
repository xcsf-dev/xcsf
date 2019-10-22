/*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
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
#include "cl_set.h"
#include "ea.h"

double xcsf_learn_trial(XCSF *xcsf, double *pred, double *x, double *y);
double xcsf_test_trial(XCSF *xcsf, double *pred, double *x, double *y);
size_t xcsf_load_params(XCSF *xcsf, FILE *fp);
size_t xcsf_save_params(XCSF *xcsf, FILE *fp);

double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle)
{  
    gplot_init(xcsf);
    xcsf->train = true;
    double perr = 0, err = 0;
    double *pred = malloc(sizeof(double) * xcsf->num_y_vars);
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        int row = 0;
        if(shuffle) {
            row = irand_uniform(0, train_data->rows);
        }
        else {
            row = (cnt % train_data->rows + train_data->rows) % train_data->rows;
        }
        double *x = &train_data->x[row * train_data->x_cols];
        double *y = &train_data->y[row * train_data->y_cols];
        double error = xcsf_learn_trial(xcsf, pred, x, y);
        perr += error;
        err += error;
        if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf1(xcsf, perr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0;
        }
    }
    free(pred);
    gplot_free(xcsf);
    return err/xcsf->MAX_TRIALS;
}

double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle)
{   
    gplot_init(xcsf);
    double perr = 0, err = 0, pterr = 0;
    double *pred = malloc(sizeof(double) * xcsf->num_y_vars);
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        int row = 0;
        if(shuffle) {
            row = irand_uniform(0, train_data->rows);
        }
        else {
            row = (cnt % train_data->rows + train_data->rows) % train_data->rows;
        }     	
        double *x = &train_data->x[row * train_data->x_cols];
        double *y = &train_data->y[row * train_data->y_cols];
        xcsf->train = true;
        double error = xcsf_learn_trial(xcsf, pred, x, y);
        perr += error; 
        err += error;
        row = irand_uniform(0, test_data->rows);
        if(shuffle) {
            row = irand_uniform(0, test_data->rows);
        }
        else {
            row = (cnt % test_data->rows + test_data->rows) % test_data->rows;
        }     	
        x = &test_data->x[row * test_data->x_cols];
        y = &test_data->y[row * test_data->y_cols];
        xcsf->train = false;
        pterr += xcsf_test_trial(xcsf, pred, x, y);
        if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf2(xcsf, perr/xcsf->PERF_AVG_TRIALS, pterr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0; pterr = 0;
        }
    }
    free(pred);
    gplot_free(xcsf);
    return err/xcsf->MAX_TRIALS;
}

double xcsf_learn_trial(XCSF *xcsf, double *pred, double *x, double *y)
{
    SET mset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    set_pred(xcsf, &mset, x, pred);
    set_update(xcsf, &mset, &kset, x, y);
    ea(xcsf, &mset, &kset);
    xcsf->time += 1;
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    set_kill(xcsf, &kset); // kills deleted classifiers
    set_free(xcsf, &mset); // frees the match set list
    return (xcsf->loss_ptr)(xcsf, pred, y);
}

double xcsf_test_trial(XCSF *xcsf, double *pred, double *x, double *y)
{
    SET mset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    set_pred(xcsf, &mset, x, pred);
    xcsf->msetsize += (xcsf->msetsize - mset.size) * xcsf->BETA;
    set_kill(xcsf, &kset); // kills deleted classifiers
    set_free(xcsf, &mset); // frees the match set list  
    return (xcsf->loss_ptr)(xcsf, pred, y);
}

void xcsf_predict(XCSF *xcsf, double *input, double *output, int rows)
{   
    xcsf->train = false;
    for(int row = 0; row < rows; row++) {
        SET mset, kset;
        set_init(xcsf, &mset);
        set_init(xcsf, &kset);
        set_match(xcsf, &mset, &kset, &input[row * xcsf->num_x_vars]);
        set_pred(xcsf, &mset, &input[row * xcsf->num_x_vars], &output[row * xcsf->num_y_vars]);
        set_kill(xcsf, &kset); // kills deleted classifiers
        set_free(xcsf, &mset); // frees the match set list
    }
}

void xcsf_print_pop(XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp)
{
    set_print(xcsf, &xcsf->pset, printc, printa, printp);
}

void xcsf_print_match_set(XCSF *xcsf, double *input, _Bool printc, _Bool printa, _Bool printp)
{
    SET mset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, input);
    set_print(xcsf, &mset, printc, printa, printp);
    set_kill(xcsf, &kset); // kills deleted classifiers
    set_free(xcsf, &mset); // frees the match set list
}

double xcsf_version()
{
    return 1.00;
}

size_t xcsf_save(XCSF *xcsf, char *fname)
{
    FILE *fp = fopen(fname, "wb");
    if(fp == 0) {
        printf("Error opening save file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0;
    double version = xcsf_version();
    s += fwrite(&version, sizeof(double), 1, fp);
    s += xcsf_save_params(xcsf, fp);
    s += pop_save(xcsf, fp);
    fclose(fp);
    //printf("xcsf saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t xcsf_load(XCSF *xcsf, char *fname)
{
    FILE *fp = fopen(fname, "rb");
    if(fp == 0) {
        printf("Error opening load file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0; double version = 0;
    s += fread(&version, sizeof(double), 1, fp);
    if(version != xcsf_version()) {
        printf("Error loading file: %s. Version mismatch. ", fname);
        printf("This version: %f. ", xcsf_version());
        printf("Loaded version: %f.\n", version);
        exit(EXIT_FAILURE);
    }
    if(xcsf->pset.size > 0) {
        set_kill(xcsf, &xcsf->pset);
        set_init(xcsf, &xcsf->pset);
    }
    s += xcsf_load_params(xcsf, fp);
    s += pop_load(xcsf, fp);
    fclose(fp);
    //printf("xcsf loaded %lu elements\n", (unsigned long)s);
    return s;
}

size_t xcsf_save_params(XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->time, sizeof(int), 1, fp);
    s += fwrite(&xcsf->msetsize, sizeof(double), 1, fp);
    s += fwrite(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->THETA_MNA, sizeof(int), 1, fp);
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
    s += fwrite(&xcsf->THETA_DEL, sizeof(double), 1, fp);
    s += fwrite(&xcsf->COND_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PRED_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ACT_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fwrite(&xcsf->P_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->F_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->S_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->E_MUTATION, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_OFFSPRING, sizeof(int), 1, fp);
    s += fwrite(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->SAM_NUM, sizeof(int), 1, fp);
    s += fwrite(&xcsf->SAM_MIN, sizeof(double), 1, fp);
    s += fwrite(&xcsf->MAX_CON, sizeof(double), 1, fp);
    s += fwrite(&xcsf->MIN_CON, sizeof(double), 1, fp);
    s += fwrite(&xcsf->NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->MOMENTUM, sizeof(double), 1, fp);
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
    s += fwrite(&xcsf->MAX_FORWARD, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->X0, sizeof(double), 1, fp);
    s += fwrite(&xcsf->RLS_SCALE_FACTOR, sizeof(double), 1, fp);
    s += fwrite(&xcsf->RLS_LAMBDA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->EA_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->SET_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->THETA_SUB, sizeof(double), 1, fp);
//    s += fwrite(&xcsf->stage, sizeof(int), 1, fp);
    s += fwrite(&xcsf->train, sizeof(_Bool), 1, fp);
    s += fwrite(&xcsf->num_x_vars, sizeof(int), 1, fp);
    s += fwrite(&xcsf->num_y_vars, sizeof(int), 1, fp);
    s += fwrite(&xcsf->num_classes, sizeof(int), 1, fp);
    s += fwrite(&xcsf->GP_NUM_CONS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->GP_INIT_DEPTH, sizeof(int), 1, fp);
    s += fwrite(xcsf->gp_cons, sizeof(double), xcsf->GP_NUM_CONS, fp);
    return s;
}

size_t xcsf_load_params(XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fread(&xcsf->time, sizeof(int), 1, fp);
    s += fread(&xcsf->msetsize, sizeof(double), 1, fp);
    s += fread(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_INIT, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->THETA_MNA, sizeof(int), 1, fp);
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
    s += fread(&xcsf->THETA_DEL, sizeof(double), 1, fp);
    s += fread(&xcsf->COND_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->PRED_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->ACT_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->P_CROSSOVER, sizeof(double), 1, fp);
    s += fread(&xcsf->P_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->F_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->S_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->E_MUTATION, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_EA, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_OFFSPRING, sizeof(int), 1, fp);
    s += fread(&xcsf->SAM_TYPE, sizeof(int), 1, fp);
    s += fread(&xcsf->SAM_NUM, sizeof(int), 1, fp);
    s += fread(&xcsf->SAM_MIN, sizeof(double), 1, fp);
    s += fread(&xcsf->MAX_CON, sizeof(double), 1, fp);
    s += fread(&xcsf->MIN_CON, sizeof(double), 1, fp);
    s += fread(&xcsf->NUM_HIDDEN_NEURONS, sizeof(int), 1, fp);
    s += fread(&xcsf->HIDDEN_NEURON_ACTIVATION, sizeof(int), 1, fp);
    s += fread(&xcsf->MOMENTUM, sizeof(double), 1, fp);
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
    s += fread(&xcsf->MAX_FORWARD, sizeof(int), 1, fp);
    s += fread(&xcsf->ETA, sizeof(double), 1, fp);
    s += fread(&xcsf->X0, sizeof(double), 1, fp);
    s += fread(&xcsf->RLS_SCALE_FACTOR, sizeof(double), 1, fp);
    s += fread(&xcsf->RLS_LAMBDA, sizeof(double), 1, fp);
    s += fread(&xcsf->EA_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->SET_SUBSUMPTION, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->THETA_SUB, sizeof(double), 1, fp);
//    s += fread(&xcsf->stage, sizeof(int), 1, fp);
    s += fread(&xcsf->train, sizeof(_Bool), 1, fp);
    s += fread(&xcsf->num_x_vars, sizeof(int), 1, fp);
    s += fread(&xcsf->num_y_vars, sizeof(int), 1, fp);
    s += fread(&xcsf->num_classes, sizeof(int), 1, fp);
    s += fread(&xcsf->GP_NUM_CONS, sizeof(int), 1, fp);
    s += fread(&xcsf->GP_INIT_DEPTH, sizeof(int), 1, fp);
    free(xcsf->gp_cons); // always malloced on start
    xcsf->gp_cons = malloc(sizeof(double)*xcsf->GP_NUM_CONS);
    s += fread(xcsf->gp_cons, sizeof(double), xcsf->GP_NUM_CONS, fp);
    loss_set_func(xcsf);         
    return s;
}
