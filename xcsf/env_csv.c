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

#include "env_csv.h"
#include "param.h"

#define MAX_ROWS (100000) //!< Maximum number of instances
#define MAX_COLS (200) //!< Maximum line length
#define MAX_NAME (200) //!< Maximum file name length
#define DELIM (",") //!< File delimiter

/**
 * @brief Returns the number of samples in a csv file.
 * @param [in] fin The csv file.
 * @return The number of samples.
 */
static int
env_csv_samples(FILE *fin)
{
    int n_samples = 0;
    char line[MAX_COLS];
    while (fgets(line, MAX_COLS, fin) != NULL) {
        ++n_samples;
    }
    return n_samples;
}

/**
 * @brief Returns the number of dimensions in a csv file.
 * @param [in] fin The csv file.
 * @return The number of dimensions.
 */
static int
env_csv_dim(FILE *fin)
{
    rewind(fin);
    int n_dim = 0;
    char line[MAX_COLS];
    char *saveptr = NULL;
    if (fgets(line, MAX_COLS, fin) != NULL) {
        const char *ptok = strtok_r(line, DELIM, &saveptr);
        while (ptok != NULL) {
            if (strnlen(ptok, MAX_COLS) > 0) {
                ++n_dim;
            }
            ptok = strtok_r(NULL, DELIM, &saveptr);
        }
    }
    return n_dim;
}

/**
 * @brief Reads the data from a csv file.
 * @param [in] fin The csv file.
 * @param [out] data The read data.
 * @param [in] n_samples The number of samples.
 * @param [in] n_dim The number of dimensions.
 */
static void
env_csv_read_data(FILE *fin, double **data, const int n_samples,
                  const int n_dim)
{
    rewind(fin);
    *data = malloc(sizeof(double) * n_dim * n_samples);
    char line[MAX_COLS];
    const char *str = NULL;
    char *saveptr = NULL;
    char *endptr = NULL;
    int i = 0;
    while (fgets(line, MAX_COLS, fin) != NULL && i < n_samples) {
        str = strtok_r(line, DELIM, &saveptr);
        (*data)[i * n_dim] = strtod(str, &endptr);
        for (int j = 1; j < n_dim; ++j) {
            str = strtok_r(NULL, DELIM, &saveptr);
            (*data)[i * n_dim + j] = strtod(str, &endptr);
        }
        ++i;
    }
}

/**
 * @brief Parses a specified csv file.
 * @details Provided a file name will set the data, n_samples, and n_dim.
 * @param [in] filename The name of the csv file to read.
 * @param [out] data A data structure to store the data.
 * @param [out] n_samples The number of samples in the dataset.
 * @param [out] n_dim The number of dimensions in the dataset.
 */
static void
env_csv_read(const char *filename, double **data, int *n_samples, int *n_dim)
{
    FILE *fin = fopen(filename, "rt");
    if (fin == 0) {
        printf("Error opening file: %s. %s.\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
    *n_samples = env_csv_samples(fin);
    *n_dim = env_csv_dim(fin);
    if (*n_samples > 0 && *n_dim > 0) {
        env_csv_read_data(fin, data, *n_samples, *n_dim);
        fclose(fin);
    } else {
        printf("Error reading file: %s. No samples found\n", filename);
        fclose(fin);
        exit(EXIT_FAILURE);
    }
    printf("Loaded: %s: samples=%d, dim=%d\n", filename, *n_samples, *n_dim);
}

/**
 * @brief Parses specified csv files into training and testing data sets.
 * @pre Identical number of x and y samples.
 * @param [in] infile The base name of the csv files to read.
 * @param [out] train_data The data structure to load the training data.
 * @param [out] test_data The data structure to load the testing data.
 */
static void
env_csv_input_read(const char *infile, struct Input *train_data,
                   struct Input *test_data)
{
    char name[MAX_NAME];
    snprintf(name, MAX_NAME, "%s_train_x.csv", infile);
    env_csv_read(name, &train_data->x, &train_data->n_samples,
                 &train_data->x_dim);
    snprintf(name, MAX_NAME, "%s_train_y.csv", infile);
    env_csv_read(name, &train_data->y, &train_data->n_samples,
                 &train_data->y_dim);
    snprintf(name, MAX_NAME, "%s_test_x.csv", infile);
    env_csv_read(name, &test_data->x, &test_data->n_samples, &test_data->x_dim);
    snprintf(name, MAX_NAME, "%s_test_y.csv", infile);
    env_csv_read(name, &test_data->y, &test_data->n_samples, &test_data->y_dim);
}

/**
 * @brief Initialises a CSV input environment from a specified filename.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] filename The file name of the csv data.
 */
void
env_csv_init(struct XCSF *xcsf, const char *filename)
{
    struct EnvCSV *env = malloc(sizeof(struct EnvCSV));
    env->train_data = malloc(sizeof(struct Input));
    env->test_data = malloc(sizeof(struct Input));
    env_csv_input_read(filename, env->train_data, env->test_data);
    xcsf->env = env;
    const int x_dim = env->train_data->x_dim;
    const int y_dim = env->train_data->y_dim;
    param_init(xcsf, x_dim, y_dim, 1);
}

/**
 * @brief Frees the csv environment.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_csv_free(const struct XCSF *xcsf)
{
    struct EnvCSV *env = xcsf->env;
    free(env->train_data->x);
    free(env->train_data->y);
    free(env->test_data->x);
    free(env->test_data->y);
    free(env->train_data);
    free(env->test_data);
    free(env);
}

/**
 * @brief Dummy method since no csv environment reset is necessary.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_csv_reset(const struct XCSF *xcsf)
{
    (void) xcsf;
}

/**
 * @brief Returns whether the csv environment is in a terminal state.
 * @param [in] xcsf The XCSF data structure.
 * @return True.
 */
bool
env_csv_is_done(const struct XCSF *xcsf)
{
    (void) xcsf;
    return true;
}

/**
 * @brief Dummy method since no state is returned by the csv environment.
 * @param [in] xcsf The XCSF data structure.
 * @return 0.
 */
const double *
env_csv_get_state(const struct XCSF *xcsf)
{
    (void) xcsf;
    return 0;
}

/**
 * @brief Dummy method since no action is executed by the csv environment.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The selected action.
 * @return 0.
 */
double
env_csv_execute(const struct XCSF *xcsf, const int action)
{
    (void) xcsf;
    (void) action;
    return 0;
}

/**
 * @brief Returns whether the csv environment is a multistep problem.
 * @param [in] xcsf The XCSF data structure.
 * @return False.
 */
bool
env_csv_multistep(const struct XCSF *xcsf)
{
    (void) xcsf;
    return false;
}

/**
 * @brief Returns the maximum payoff value possible in the csv environment.
 * @param [in] xcsf The XCSF data structure.
 * @return 0.
 */
double
env_csv_maxpayoff(const struct XCSF *xcsf)
{
    (void) xcsf;
    return 0;
}
