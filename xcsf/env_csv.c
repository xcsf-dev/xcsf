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
 * @file env_csv.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief CSV input file handling functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include "xcsf.h"
#include "utils.h"
#include "env.h"
#include "env_csv.h"

#define MAX_ROWS 100000 //!< Maximum number of instances
#define MAX_COLS 200 //!< Maximum line length
#define MAX_NAME 200 //!< Maximum file name length
#define DELIM "," //!< File delimiter

static void env_csv_read(const char *fname, double **data, int *num_prob, int *num_vars);
static void env_csv_input_read(const char *infile, INPUT *train_data, INPUT *test_data);

/**
 * @brief Initialises a CSV input environment from a specified filename.
 * @param xcsf The XCSF data structure.
 * @param fname The file name of the csv data.
 */
void env_csv_init(XCSF *xcsf, const char *fname)
{
    ENV_CSV *env = malloc(sizeof(ENV_CSV));
    env->train_data = malloc(sizeof(INPUT));
    env->test_data = malloc(sizeof(INPUT));
    env_csv_input_read(fname, env->train_data, env->test_data);
    xcsf->env = env;
    xcsf->x_dim = env->train_data->x_dim;
    xcsf->y_dim = env->train_data->y_dim;
    xcsf->n_actions = 1;
}

/**
 * @brief Frees the csv environment.
 * @param xcsf The XCSF data structure.
 */
void env_csv_free(const XCSF *xcsf)
{
    ENV_CSV *env = xcsf->env;
    free(env->train_data->x);
    free(env->train_data->y);
    free(env->test_data->x);
    free(env->test_data->y);
    free(env->train_data);
    free(env->test_data);
    free(env);
}

/**
 * @brief Parses specified csv files into training and testing data sets.
 * @param infile The base name of the csv files to read.
 * @param train_data The data structure to load the training data.
 * @param test_data The data structure to load the testing data.
 *
 * @details Expects an identical number of x and y samples.
 */
static void env_csv_input_read(const char *infile, INPUT *train_data, INPUT *test_data)
{
    char name[MAX_NAME];
    snprintf(name, MAX_NAME, "%s_train_x.csv", infile);
    env_csv_read(name, &train_data->x, &train_data->n_samples, &train_data->x_dim);
    snprintf(name, MAX_NAME, "%s_train_y.csv", infile);
    env_csv_read(name, &train_data->y, &train_data->n_samples, &train_data->y_dim);
    snprintf(name, MAX_NAME, "%s_test_x.csv", infile);
    env_csv_read(name, &test_data->x, &test_data->n_samples, &test_data->x_dim);
    snprintf(name, MAX_NAME, "%s_test_y.csv", infile);
    env_csv_read(name, &test_data->y, &test_data->n_samples, &test_data->y_dim);
}

/**
 * @brief Parses a specified csv file.
 * @param fname The name of the csv file to read.
 * @param data The data structure to load the data (set by this function).
 * @param n_samples The number of samples in the dataset (set by this function).
 * @param dim The number of dimensions in the dataset (set by this function).
 *
 * @details Provided a file name will set the data, n_samples, and dim.
 */
static void env_csv_read(const char *fname, double **data, int *n_samples, int *dim)
{
    FILE *fin = fopen(fname, "rt");
    if(fin == 0) {
        printf("Error opening file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }    
    // ascertain the file length and number of variables per line
    *n_samples = 0;
    *dim = 0;
    char line[MAX_COLS];
    char *saveptr;
    while(fgets(line, MAX_COLS, fin) != NULL) {
        if(*n_samples > MAX_ROWS) {
            printf("data file %s is too big; maximum: %d\n", fname, MAX_ROWS);
            exit(EXIT_FAILURE);
        }        
        // use the first line to count the number of variables on a line
        if(*n_samples == 0) {
            const char *ptok = strtok_r(line, DELIM, &saveptr);
            while(ptok != NULL) {
                if(strnlen(ptok,MAX_COLS) > 0) {
                    (*dim)++;
                }
                ptok = strtok_r(NULL, DELIM, &saveptr);
            }
        }
        (*n_samples)++; // count the number of lines
    }
    // read data file to memory
    if(*dim > 0 && *n_samples > 0) {
        rewind(fin);
        *data = malloc(sizeof(double) * (*dim) * (*n_samples));
        int i = 0;
        while(fgets(line,MAX_COLS,fin) != NULL) {
            (*data)[i * (*dim)] = atof(strtok_r(line, DELIM, &saveptr));
            for(int j = 1; j < *dim; j++) {
                (*data)[i * (*dim)+j] = atof(strtok_r(NULL, DELIM, &saveptr));
            }
            i++;
        }
    }
    fclose(fin);
    printf("Loaded: %s: %d samples, %d dimensions\n", fname, *n_samples, *dim);
}

void env_csv_reset(const XCSF *xcsf)
{
    (void)xcsf;
}

_Bool env_csv_isreset(const XCSF *xcsf)
{
    (void)xcsf;
    return true;
}

const double *env_csv_get_state(const XCSF *xcsf)
{
    (void)xcsf;
    return 0;
}

double env_csv_execute(const XCSF *xcsf, int action)
{
    (void)xcsf;
    (void)action;
    return 0;
}

_Bool env_csv_multistep(const XCSF *xcsf)
{
    (void)xcsf;
    return false;
}

double env_csv_maxpayoff(const XCSF *xcsf)
{
    (void)xcsf;
    return 0;
}
