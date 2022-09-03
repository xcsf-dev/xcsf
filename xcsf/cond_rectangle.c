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
 * @file cond_rectangle.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2022.
 * @brief Hyperrectangle condition functions.
 */

#include "cond_rectangle.h"
#include "ea.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of hyperrectangle mutation rates

/**
 * @brief Self-adaptation method for mutating hyperrectangles.
 */
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL };

/**
 * @brief Returns the relative distance to a hyperrectangle.
 * @details Distance is zero at the center; one on the border; and greater than
 * one outside of the hyperrectangle.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose hyperrectangle distance is to be computed.
 * @param [in] x Input to compute the relative distance.
 * @return The relative distance of an input to the hyperrectangle.
 */
static double
cond_rectangle_dist(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    const struct CondRectangle *cond = c->cond;
    double dist = 0;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        const double d = fabs((x[i] - cond->b1[i]) / cond->b2[i]);
        if (d > dist) {
            dist = d;
        }
    }
    return dist;
}

/**
 * @brief Creates and initialises a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 */
void
cond_rectangle_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        new->b1[i] = rand_uniform(xcsf->cond->min, xcsf->cond->max);
        new->b2[i] = rand_uniform(xcsf->cond->min, xcsf->cond->max);
    }
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        // csr: b1 = center, b2 = spread
        for (int i = 0; i < xcsf->x_dim; ++i) {
            new->b2[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
        }
    }
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_rectangle_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondRectangle *cond = c->cond;
    free(cond->b1);
    free(cond->b2);
    free(cond->mu);
    free(c->cond);
}

/**
 * @brief Copies a hyperrectangle condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_rectangle_copy(const struct XCSF *xcsf, struct Cl *dest,
                    const struct Cl *src)
{
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    const struct CondRectangle *src_cond = src->cond;
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->b1, src_cond->b1, sizeof(double) * xcsf->x_dim);
    memcpy(new->b2, src_cond->b2, sizeof(double) * xcsf->x_dim);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

/**
 * @brief Generates a hyperrectangle that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_rectangle_cover(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    const struct CondRectangle *cond = c->cond;
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->b1[i] = x[i];
            cond->b2[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
        }
    } else {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            const double r1 = rand_uniform(xcsf->cond->spread_min, spread_max);
            const double r2 = rand_uniform(xcsf->cond->spread_min, spread_max);
            cond->b1[i] = x[i] - (r1 * 0.5);
            cond->b2[i] = x[i] + (r2 * 0.5);
        }
    }
}

/**
 * @brief Updates a hyperrectangle, sliding the centers towards the mean input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_rectangle_update(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x, const double *y)
{
    (void) y;
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR &&
        xcsf->cond->eta > 0) {
        const struct CondRectangle *cond = c->cond;
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->b1[i] += xcsf->cond->eta * (x[i] - cond->b1[i]);
        }
    }
}

/**
 * @brief Calculates whether a hyperrectangle condition matches an input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return Whether the hyperrectangle condition matches the input.
 */
bool
cond_rectangle_match(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    const struct CondRectangle *cond = c->cond;
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        return (cond_rectangle_dist(xcsf, c, x) < 1);
    } else { // ubr
        for (int i = 0; i < xcsf->x_dim; ++i) {
            const double lb = fmin(cond->b1[i], cond->b2[i]);
            const double ub = fmax(cond->b1[i], cond->b2[i]);
            if (x[i] < lb || x[i] > ub) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Performs uniform crossover with two hyperrectangle conditions.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
bool
cond_rectangle_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                         const struct Cl *c2)
{
    const struct CondRectangle *cond1 = c1->cond;
    const struct CondRectangle *cond2 = c2->cond;
    bool changed = false;
    if (rand_uniform(0, 1) < xcsf->ea->p_crossover) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->b1[i];
                cond1->b1[i] = cond2->b1[i];
                cond2->b1[i] = tmp;
                changed = true;
            }
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->b2[i];
                cond1->b2[i] = cond2->b2[i];
                cond2->b2[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Mutates a hyperrectangle condition with the self-adaptive rate.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_rectangle_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    bool changed = false;
    const struct CondRectangle *cond = c->cond;
    double *b1 = cond->b1;
    double *b2 = cond->b2;
    sam_adapt(cond->mu, N_MU, MU_TYPE);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double orig = b1[i];
        b1[i] += rand_normal(0, cond->mu[0]);
        b1[i] = clamp(b1[i], xcsf->cond->min, xcsf->cond->max);
        if (orig != b1[i]) {
            changed = true;
        }
        orig = b2[i];
        b2[i] += rand_normal(0, cond->mu[0]);
        if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
            b2[i] = fmax(DBL_EPSILON, b2[i]);
        } else {
            b2[i] = clamp(b2[i], xcsf->cond->min, xcsf->cond->max);
        }
        if (orig != b2[i]) {
            changed = true;
        }
    }
    return changed;
}

/**
 * @brief Returns whether classifier c1 has a condition more general than c2.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return Whether the hyperrectangle condition of c1 is more general than c2.
 */
bool
cond_rectangle_general(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2)
{
    const struct CondRectangle *cond1 = c1->cond;
    const struct CondRectangle *cond2 = c2->cond;
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            const double l1 = cond1->b1[i] - cond1->b2[i];
            const double l2 = cond2->b1[i] - cond2->b2[i];
            const double u1 = cond1->b1[i] + cond1->b2[i];
            const double u2 = cond2->b1[i] + cond2->b2[i];
            if (l1 > l2 || u1 < u2) {
                return false;
            }
        }
    } else {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            const double l1 = fmin(cond1->b1[i], cond1->b2[i]);
            const double l2 = fmin(cond2->b1[i], cond2->b2[i]);
            const double u1 = fmax(cond1->b1[i], cond1->b2[i]);
            const double u2 = fmax(cond2->b1[i], cond2->b2[i]);
            if (l1 > l2 || u1 < u2) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Prints a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_rectangle_print(const struct XCSF *xcsf, const struct Cl *c)
{
    char *json_str = cond_rectangle_json_export(xcsf, c);
    printf("%s\n", json_str);
    free(json_str);
}

/**
 * @brief Returns the size of a hyperrectangle condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The length of the input dimension.
 */
double
cond_rectangle_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) c;
    return xcsf->x_dim;
}

/**
 * @brief Writes a hyperrectangle condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_rectangle_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    size_t s = 0;
    const struct CondRectangle *cond = c->cond;
    s += fwrite(cond->b1, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->b2, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a hyperrectangle condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_rectangle_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    size_t s = 0;
    struct CondRectangle *new = malloc(sizeof(struct CondRectangle));
    new->b1 = malloc(sizeof(double) * xcsf->x_dim);
    new->b2 = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->b1, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->b2, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a hyperrectangle.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
char *
cond_rectangle_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondRectangle *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON *b1 = cJSON_CreateDoubleArray(cond->b1, xcsf->x_dim);
    cJSON *b2 = cJSON_CreateDoubleArray(cond->b2, xcsf->x_dim);
    cJSON *mutation = cJSON_CreateDoubleArray(cond->mu, N_MU);
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        cJSON_AddStringToObject(json, "type", "hyperrectangle_csr");
        cJSON_AddItemToObject(json, "center", b1);
        cJSON_AddItemToObject(json, "spread", b2);
    } else {
        cJSON_AddStringToObject(json, "type", "hyperrectangle_ubr");
        cJSON_AddItemToObject(json, "bound1", b1);
        cJSON_AddItemToObject(json, "bound2", b2);
    }
    cJSON_AddItemToObject(json, "mutation", mutation);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Creates a hyperrectangle from a cJSON object.
 * @param [in] xcsf The XCSF data structure.
 * @param [in,out] c The classifier to initialise.
 * @param [in] json cJSON object.
 */
void
cond_rectangle_json_import(const struct XCSF *xcsf, struct Cl *c,
                           const cJSON *json)
{
    struct CondRectangle *cond = c->cond;
    bool csr = false;
    if (xcsf->cond->type == COND_TYPE_HYPERRECTANGLE_CSR) {
        csr = true;
    }
    const char *b1_name = csr ? "center" : "bound1";
    const char *b2_name = csr ? "spread" : "bound2";
    const cJSON *item = cJSON_GetObjectItem(json, b1_name);
    if (item != NULL && cJSON_IsArray(item)) {
        if (cJSON_GetArraySize(item) == xcsf->x_dim) {
            for (int i = 0; i < xcsf->x_dim; ++i) {
                const cJSON *item_i = cJSON_GetArrayItem(item, i);
                cond->b1[i] = item_i->valuedouble;
            }
        } else {
            printf("Import error: %s length mismatch\n", b1_name);
            exit(EXIT_FAILURE);
        }
    }
    item = cJSON_GetObjectItem(json, b2_name);
    if (item != NULL && cJSON_IsArray(item)) {
        if (cJSON_GetArraySize(item) == xcsf->x_dim) {
            for (int i = 0; i < xcsf->x_dim; ++i) {
                const cJSON *item_i = cJSON_GetArrayItem(item, i);
                cond->b2[i] = item_i->valuedouble;
            }
        } else {
            printf("Import error: %s length mismatch\n", b2_name);
            exit(EXIT_FAILURE);
        }
    }
    sam_json_import(cond->mu, N_MU, json);
}
