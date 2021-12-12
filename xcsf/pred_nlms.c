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
 * @file pred_nlms.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Normalised least mean squares prediction functions.
 */

#include "pred_nlms.h"
#include "blas.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of self-adaptive mutation rates

/**
 * @brief Self-adaptation method for mutating NLMS predictions.
 */
static const int MU_TYPE[N_MU] = {
    SAM_LOG_NORMAL //!< Rate of gradient descent mutation
};

/**
 * @brief Initialises an NLMS prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be initialised.
 */
void
pred_nlms_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct PredNLMS *pred = malloc(sizeof(struct PredNLMS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if (xcsf->pred->type == PRED_TYPE_NLMS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->n = 1 + 2 * xcsf->x_dim + xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    } else {
        pred->n = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->n_weights = pred->n * xcsf->y_dim;
    pred->weights = calloc(pred->n_weights, sizeof(double));
    blas_fill(xcsf->y_dim, xcsf->pred->x0, pred->weights, pred->n);
    // initialise learning rate
    pred->mu = malloc(sizeof(double) * N_MU);
    if (xcsf->pred->evolve_eta) {
        sam_init(pred->mu, N_MU, MU_TYPE);
        pred->eta = rand_uniform(xcsf->pred->eta_min, xcsf->pred->eta);
    } else {
        memset(pred->mu, 0, sizeof(double) * N_MU);
        pred->eta = xcsf->pred->eta;
    }
    // initialise temporary storage for weight updating
    pred->tmp_input = malloc(sizeof(double) * pred->n);
}

/**
 * @brief Copies an NLMS prediction from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
pred_nlms_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    pred_nlms_init(xcsf, dest);
    struct PredNLMS *dest_pred = dest->pred;
    const struct PredNLMS *src_pred = src->pred;
    memcpy(dest_pred->weights, src_pred->weights,
           sizeof(double) * src_pred->n_weights);
    memcpy(dest_pred->mu, src_pred->mu, sizeof(double) * N_MU);
    dest_pred->eta = src_pred->eta;
}

/**
 * @brief Frees the memory used by an NLMS prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be freed.
 */
void
pred_nlms_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct PredNLMS *pred = c->pred;
    free(pred->weights);
    free(pred->tmp_input);
    free(pred->mu);
    free(pred);
}

/**
 * @brief Updates an NLMS prediction for a given input and truth sample.
 * @pre The prediction has been computed for the current state.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose prediction is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
pred_nlms_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const double *y)
{
    const struct PredNLMS *pred = c->pred;
    // normalise update
    const int n = pred->n;
    const double X0 = xcsf->pred->x0;
    const double norm = X0 * X0 + blas_dot(xcsf->x_dim, x, 1, x, 1);
    // update weights using the error
    for (int i = 0; i < xcsf->y_dim; ++i) {
        const double error = y[i] - c->prediction[i];
        const double correction = (pred->eta * error) / norm;
        blas_axpy(n, correction, pred->tmp_input, 1, &pred->weights[i * n], 1);
    }
}

/**
 * @brief Computes the current NLMS prediction for a provided input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the prediction.
 * @param [in] x The input state.
 */
void
pred_nlms_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct PredNLMS *pred = c->pred;
    const int n = pred->n;
    pred_transform_input(xcsf, x, xcsf->pred->x0, pred->tmp_input);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        c->prediction[i] =
            blas_dot(n, &pred->weights[i * n], 1, pred->tmp_input, 1);
    }
}

/**
 * @brief Prints an NLMS prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be printed.
 */
void
pred_nlms_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", pred_nlms_json_export(xcsf, c));
}

/**
 * @brief Dummy function since NLMS predictions do not perform crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose prediction is being crossed.
 * @param [in] c2 The second classifier whose prediction is being crossed.
 * @return False.
 */
bool
pred_nlms_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Mutates the gradient descent rate used to update an NLMS prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is being mutated.
 * @return Whether any alterations were made.
 */
bool
pred_nlms_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    if (xcsf->pred->evolve_eta) {
        struct PredNLMS *pred = c->pred;
        sam_adapt(pred->mu, N_MU, MU_TYPE);
        const double orig = pred->eta;
        pred->eta += rand_normal(0, pred->mu[0]);
        pred->eta = clamp(pred->eta, xcsf->pred->eta_min, xcsf->pred->eta);
        if (orig != pred->eta) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Returns the size of an NLMS prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction size to return.
 * @return The number of weights.
 */
double
pred_nlms_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNLMS *pred = c->pred;
    return pred->n_weights;
}

/**
 * @brief Writes an NLMS prediction to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
pred_nlms_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct PredNLMS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->n, sizeof(int), 1, fp);
    s += fwrite(&pred->n_weights, sizeof(int), 1, fp);
    s += fwrite(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fwrite(pred->mu, sizeof(double), N_MU, fp);
    s += fwrite(&pred->eta, sizeof(double), 1, fp);
    return s;
}

/**
 * @brief Reads an NLMS prediction from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
pred_nlms_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    pred_nlms_init(xcsf, c);
    struct PredNLMS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->n, sizeof(int), 1, fp);
    s += fread(&pred->n_weights, sizeof(int), 1, fp);
    s += fread(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fread(pred->mu, sizeof(double), N_MU, fp);
    s += fread(&pred->eta, sizeof(double), 1, fp);
    return s;
}

/**
 * @brief Returns a json formatted string representation of an NLMS prediction.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose prediction is to be returned.
 * @return String encoded in json format.
 */
const char *
pred_nlms_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct PredNLMS *pred = c->pred;
    cJSON *json = cJSON_CreateObject();
    if (xcsf->pred->type == PRED_TYPE_NLMS_QUADRATIC) {
        cJSON_AddStringToObject(json, "type", "nlms_quadratic");
    } else {
        cJSON_AddStringToObject(json, "type", "nlms_linear");
    }
    cJSON *weights = cJSON_CreateDoubleArray(pred->weights, pred->n_weights);
    cJSON_AddItemToObject(json, "weights", weights);
    cJSON_AddNumberToObject(json, "eta", pred->eta);
    cJSON *mutation = cJSON_CreateDoubleArray(pred->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
