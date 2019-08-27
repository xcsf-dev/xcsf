/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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
 **************
 * Description: 
 **************
 * The main XCSF module.  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "xcsf.h"
#include "config.h"
#include "utils.h"
#include "loss.h"
#include "input.h"
#include "perf.h"
#include "cl.h"
#include "cl_set.h"
#include "ga.h"

#ifdef PARALLEL
#include <omp.h>
#endif

double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle);
double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle);
void xcsf_predict(XCSF *xcsf, double *input, double *output, int rows);
double xcsf_learn_trial(XCSF *xcsf, double *pred, double *x, double *y);
double xcsf_test_trial(XCSF *xcsf, double *pred, double *x, double *y);

int main(int argc, char **argv)
{    
    if(argc < 2 || argc > 3) {
        printf("Usage: xcsf inputfile [config.ini]\n");
        exit(EXIT_FAILURE);
    } 

    XCSF *xcsf = malloc(sizeof(XCSF));
    random_init();
    if(argc > 2) {
        constants_init(xcsf, argv[2]);
    }    
    else {
        constants_init(xcsf, "default.ini");
    }
#ifdef PARALLEL
    omp_set_num_threads(xcsf->OMP_NUM_THREADS);
#endif

    INPUT *train_data = malloc(sizeof(INPUT));
    INPUT *test_data = malloc(sizeof(INPUT));
    input_read_csv(argv[1], train_data, test_data);
    xcsf->num_x_vars = train_data->x_cols;
    xcsf->num_y_vars = train_data->y_cols;
    xcsf->num_classes = 0; // regression

    pop_init(xcsf);
    xcsf_fit2(xcsf, train_data, test_data, true);

    set_kill(xcsf, &xcsf->pset);
    constants_free(xcsf);        
    free(xcsf);
    input_free(train_data);
    input_free(test_data);
    free(train_data);
    free(test_data);
    return EXIT_SUCCESS;
}

double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle)
{  
#ifdef GNUPLOT
    gplot_init(xcsf);
#endif

    xcsf->train = true;
    double perr = 0, err = 0;
    double *pred = malloc(sizeof(double)*xcsf->num_y_vars);
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

#ifdef GNUPLOT
    gplot_free(xcsf);
#endif

    return err/xcsf->MAX_TRIALS;
}

double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle)
{   
#ifdef GNUPLOT
    gplot_init(xcsf);
#endif

    double perr = 0, err = 0, pterr = 0;
    double *pred = malloc(sizeof(double)*xcsf->num_y_vars);
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

#ifdef GNUPLOT
    gplot_free(xcsf);
#endif

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
    ga(xcsf, &mset, &kset);
    xcsf->time += 1;
    xcsf->msetsize += (mset.size - xcsf->msetsize)*xcsf->BETA;
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
    xcsf->msetsize += (xcsf->msetsize - mset.size)*xcsf->BETA;
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
        set_match(xcsf, &mset, &kset, &input[row*xcsf->num_x_vars]);
        set_pred(xcsf, &mset, &input[row*xcsf->num_x_vars], &output[row*xcsf->num_y_vars]);
        set_kill(xcsf, &kset); // kills deleted classifiers
        set_free(xcsf, &mset); // frees the match set list      
    }
}

void xcsf_print_pop(XCSF *xcsf, _Bool print_cond, _Bool print_pred)
{
    set_print(xcsf, &xcsf->pset, print_cond, print_pred);
}

void xcsf_print_match_set(XCSF *xcsf, double *input, _Bool print_cond, _Bool print_pred)
{
    SET mset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, input);
    set_print(xcsf, &mset, print_cond, print_pred);
}
