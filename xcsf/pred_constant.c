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
 * @file pred_constant.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Piece-wise constant prediction functions.
 */

#include "pred_constant.h"
#include "utils.h"

/**
 * @brief Dummy function since constant predictions have no data structure.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be initialised.
 */
void
pred_constant_init(const struct XCSF *xcsf, struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

/**
 * @brief Dummy function since constant predictions have no data structure.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
pred_constant_copy(const struct XCSF *xcsf, struct Cl *dest,
                   const struct Cl *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

/**
 * @brief Dummy function since constant predictions have no data structure.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be freed.
 */
void
pred_constant_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

/**
 * @brief Updates a constant prediction for a given input and truth sample.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose prediction is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
pred_constant_update(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x, const double *y)
{
    (void) x;
    if (c->exp * xcsf->BETA < 1) {
        for (int i = 0; i < xcsf->y_dim; ++i) {
            c->prediction[i] =
                (c->prediction[i] * (c->exp - 1) + y[i]) / c->exp;
        }
    } else {
        for (int i = 0; i < xcsf->y_dim; ++i) {
            c->prediction[i] += xcsf->BETA * (y[i] - c->prediction[i]);
        }
    }
}

/**
 * @brief Dummy function since constant predictions are not computed.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the prediction.
 * @param [in] x The input state.
 */
void
pred_constant_compute(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

/**
 * @brief Prints a constant prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be printed.
 */
void
pred_constant_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", pred_constant_json_export(xcsf, c));
}

/**
 * @brief Dummy function since constant predictions do not perform crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose prediction is being crossed.
 * @param [in] c2 The second classifier whose prediction is being crossed.
 * @return False.
 */
bool
pred_constant_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                        const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy function since constant predictions do not perform mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is being mutated.
 * @return False.
 */
bool
pred_constant_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

/**
 * @brief Returns the size of a constant prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction size to return.
 * @return The output dimension.
 */
double
pred_constant_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) c;
    return xcsf->y_dim;
}

/**
 * @brief Dummy function since constant predictions have no data structure.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return 0.
 */
size_t
pred_constant_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

/**
 * @brief Dummy function since constant predictions have no data structure.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return 0.
 */
size_t
pred_constant_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

/**
 * @brief Returns a json formatted string representation of a prediction.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose prediction is to be returned.
 * @return String encoded in json format.
 */
const char *
pred_constant_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "constant");
    cJSON *prediction = cJSON_CreateDoubleArray(c->prediction, xcsf->y_dim);
    cJSON_AddItemToObject(json, "prediction", prediction);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
