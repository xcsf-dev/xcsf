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
#include "param.h"
#include "clset.h"
#include "cl.h"
#include "neural.h"
#include "prediction.h"
#include "condition.h"
#include "cond_neural.h"
#include "pred_neural.h"

static const double VERSION = 1.06; //!< XCSF version number

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
    s += param_save(xcsf, fp);
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
    s += param_load(xcsf, fp);
    s += clset_pop_load(xcsf, fp);
    fclose(fp);
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
    param_set_y_dim(xcsf, y_dim);
    param_set_auto_encode(xcsf, false);
    param_set_loss_func(xcsf, 5); // one-hot encoding error
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
        const CL *src = iter->cl;
        cl_init_copy(xcsf, new, src);
        clset_add(&xcsf->prev_pset, new);
    }
}
