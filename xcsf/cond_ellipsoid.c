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
 * @file cond_ellipsoid.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2021.
 * @brief Hyperellipsoid condition functions.
 */

#include "cond_ellipsoid.h"
#include "ea.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of hyperellipsoid mutation rates

/**
 * @brief Self-adaptation method for mutating hyperellipsoids.
 */
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL };

/**
 * @brief Returns the relative distance to a hyperellipsoid.
 * @details Distance is zero at the center; one on the border; and greater than
 * one outside of the hyperellipsoid.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose hyperellipsoid distance is to be computed.
 * @param [in] x Input to compute the relative distance.
 * @return The relative distance of an input to the hyperellipsoid.
 */
static double
cond_ellipsoid_dist(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    const struct CondEllipsoid *cond = c->cond;
    double dist = 0;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        const double d = (x[i] - cond->center[i]) / cond->spread[i];
        dist += d * d;
    }
    return dist;
}

/**
 * @brief Creates and initialises a hyperellipsoid condition.
 * @details Uses the center-spread representation.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 */
void
cond_ellipsoid_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondEllipsoid *new = malloc(sizeof(struct CondEllipsoid));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        new->center[i] = rand_uniform(xcsf->cond->min, xcsf->cond->max);
        new->spread[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
    }
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a hyperellipsoid condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_ellipsoid_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondEllipsoid *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(cond->mu);
    free(c->cond);
}

/**
 * @brief Copies a hyperellipsoid condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_ellipsoid_copy(const struct XCSF *xcsf, struct Cl *dest,
                    const struct Cl *src)
{
    struct CondEllipsoid *new = malloc(sizeof(struct CondEllipsoid));
    const struct CondEllipsoid *src_cond = src->cond;
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->center, src_cond->center, sizeof(double) * xcsf->x_dim);
    memcpy(new->spread, src_cond->spread, sizeof(double) * xcsf->x_dim);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

/**
 * @brief Generates a hyperellipsoid that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_ellipsoid_cover(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    const struct CondEllipsoid *cond = c->cond;
    const double spread_max = fabs(xcsf->cond->max - xcsf->cond->min);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(xcsf->cond->spread_min, spread_max);
    }
}

/**
 * @brief Updates a hyperellipsoid, sliding the centers towards the mean input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_ellipsoid_update(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x, const double *y)
{
    (void) y;
    if (xcsf->cond->eta > 0) {
        const struct CondEllipsoid *cond = c->cond;
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->center[i] += xcsf->cond->eta * (x[i] - cond->center[i]);
        }
    }
}

/**
 * @brief Calculates whether a hyperellipsoid condition matches an input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return Whether the hyperellipsoid condition matches the input.
 */
bool
cond_ellipsoid_match(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x)
{
    return (cond_ellipsoid_dist(xcsf, c, x) < 1);
}

/**
 * @brief Performs uniform crossover with two hyperellipsoid conditions.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
bool
cond_ellipsoid_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                         const struct Cl *c2)
{
    const struct CondEllipsoid *cond1 = c1->cond;
    const struct CondEllipsoid *cond2 = c2->cond;
    bool changed = false;
    if (rand_uniform(0, 1) < xcsf->ea->p_crossover) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->center[i];
                cond1->center[i] = cond2->center[i];
                cond2->center[i] = tmp;
                changed = true;
            }
            if (rand_uniform(0, 1) < 0.5) {
                const double tmp = cond1->spread[i];
                cond1->spread[i] = cond2->spread[i];
                cond2->spread[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Mutates a hyperellipsoid condition with the self-adaptive rate.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_ellipsoid_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    bool changed = false;
    const struct CondEllipsoid *cond = c->cond;
    double *center = cond->center;
    double *spread = cond->spread;
    sam_adapt(cond->mu, N_MU, MU_TYPE);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double orig = center[i];
        center[i] += rand_normal(0, cond->mu[0]);
        center[i] = clamp(center[i], xcsf->cond->min, xcsf->cond->max);
        if (orig != center[i]) {
            changed = true;
        }
        orig = spread[i];
        spread[i] += rand_normal(0, cond->mu[0]);
        spread[i] = fmax(DBL_EPSILON, spread[i]);
        if (orig != spread[i]) {
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
 * @return Whether the hyperellipsoid condition of c1 is more general than c2.
 */
bool
cond_ellipsoid_general(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2)
{
    const struct CondEllipsoid *cond1 = c1->cond;
    const struct CondEllipsoid *cond2 = c2->cond;
    double *temp = malloc(sizeof(double) * xcsf->x_dim);
    memcpy(temp, cond2->center, sizeof(double) * xcsf->x_dim);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        if (cond1->center[i] != cond2->center[i] ||
            cond1->spread[i] != cond2->spread[i]) {
            temp[i] += cond2->spread[i];
            if (cond_ellipsoid_dist(xcsf, c1, temp) > 1) {
                free(temp);
                return false;
            }
            temp[i] -= 2 * cond2->spread[i];
            if (cond_ellipsoid_dist(xcsf, c1, temp) > 1) {
                free(temp);
                return false;
            }
            temp[i] = cond2->center[i];
        }
    }
    free(temp);
    return true;
}

/**
 * @brief Prints a hyperellipsoid condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_ellipsoid_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_ellipsoid_json_export(xcsf, c));
}

/**
 * @brief Returns the size of a hyperellipsoid condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The length of the input dimension.
 */
double
cond_ellipsoid_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) c;
    return xcsf->x_dim;
}

/**
 * @brief Writes a hyperellipsoid condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_ellipsoid_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    size_t s = 0;
    const struct CondEllipsoid *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a hyperellipsoid condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_ellipsoid_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    size_t s = 0;
    struct CondEllipsoid *new = malloc(sizeof(struct CondEllipsoid));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->center, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->spread, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a hyperellipsoid.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_ellipsoid_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondEllipsoid *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "hyperellipsoid");
    cJSON *center = cJSON_CreateDoubleArray(cond->center, xcsf->x_dim);
    cJSON *spread = cJSON_CreateDoubleArray(cond->spread, xcsf->x_dim);
    cJSON *mutation = cJSON_CreateDoubleArray(cond->mu, N_MU);
    cJSON_AddItemToObject(json, "center", center);
    cJSON_AddItemToObject(json, "spread", spread);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
