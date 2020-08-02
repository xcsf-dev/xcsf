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

#include "clset.h"
#include "cond_neural.h"
#include "pa.h"
#include "param.h"
#include "pred_neural.h"

/**
 * @brief Initialises XCSF with an empty population.
 * @param xcsf The XCSF data structure.
 */
void
xcsf_init(struct XCSF *xcsf)
{
    xcsf->time = 0;
    xcsf->msetsize = 0;
    xcsf->mfrac = 0;
    clset_init(&xcsf->pset);
}

/**
 * @brief Prints the current XCSF population.
 * @param xcsf The XCSF data structure.
 * @param printc Whether to print condition structures.
 * @param printa Whether to print action structures.
 * @param printp Whether to print prediction structures.
 */
void
xcsf_print_pop(const struct XCSF *xcsf, _Bool printc, _Bool printa,
               _Bool printp)
{
    clset_print(xcsf, &xcsf->pset, printc, printa, printp);
}

/**
 * @brief Writes the current state of XCSF to a binary file.
 * @param xcsf The XCSF data structure.
 * @param fname The name of the output file.
 * @return The total number of elements written.
 */
size_t
xcsf_save(const struct XCSF *xcsf, const char *fname)
{
    FILE *fp = fopen(fname, "wb");
    if (fp == 0) {
        printf("Error opening save file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0;
    s += fwrite(&VERSION_MAJOR, sizeof(int), 1, fp);
    s += fwrite(&VERSION_MINOR, sizeof(int), 1, fp);
    s += fwrite(&VERSION_BUILD, sizeof(int), 1, fp);
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
size_t
xcsf_load(struct XCSF *xcsf, const char *fname)
{
    if (xcsf->pset.size > 0) {
        clset_kill(xcsf, &xcsf->pset);
        clset_init(&xcsf->pset);
    }
    FILE *fp = fopen(fname, "rb");
    if (fp == 0) {
        printf("Error opening load file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t s = 0;
    int major = 0;
    int minor = 0;
    int build = 0;
    s += fread(&major, sizeof(int), 1, fp);
    s += fread(&minor, sizeof(int), 1, fp);
    s += fread(&build, sizeof(int), 1, fp);
    if (major != VERSION_MAJOR || minor != VERSION_MINOR ||
        build != VERSION_BUILD) {
        printf("Error loading file: %s. Version mismatch. ", fname);
        printf("This version: %d.%d.%d.\n", VERSION_MAJOR, VERSION_MINOR,
               VERSION_BUILD);
        printf("Loaded version: %d.%d.%d\n", major, minor, build);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    s += param_load(xcsf, fp);
    s += clset_pop_load(xcsf, fp);
    fclose(fp);
    return s;
}

/**
 * @brief Inserts a new hidden layer before the output layer within all
 * prediction neural networks in the population.
 * @param xcsf The XCSF data structure.
 */
void
xcsf_pred_expand(const struct XCSF *xcsf)
{
    for (const struct CLIST *iter = xcsf->pset.list; iter != NULL;
         iter = iter->next) {
        pred_neural_expand(xcsf, iter->cl);
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
 * @param n_del The number of hidden layers to remove.
 */
void
xcsf_ae_to_classifier(struct XCSF *xcsf, int y_dim, int n_del)
{
    pa_free(xcsf);
    param_set_y_dim(xcsf, y_dim);
    param_set_loss_func(xcsf, 5); // one-hot encoding error
    pa_init(xcsf);
    for (const struct CLIST *iter = xcsf->pset.list; iter != NULL;
         iter = iter->next) {
        free(iter->cl->prediction);
        iter->cl->prediction = calloc(xcsf->y_dim, sizeof(double));
        pred_neural_ae_to_classifier(xcsf, iter->cl, n_del);
        iter->cl->fit = xcsf->INIT_FITNESS;
        iter->cl->err = xcsf->INIT_ERROR;
        iter->cl->exp = 0;
        iter->cl->time = xcsf->time;
    }
}
