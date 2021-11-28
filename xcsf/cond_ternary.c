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
 * @file cond_ternary.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2021.
 * @brief Ternary condition functions.
 * @details Binarises inputs.
 */

#include "cond_ternary.h"
#include "ea.h"
#include "sam.h"
#include "utils.h"

#define DONT_CARE ('#') //!< Don't care symbol
#define N_MU (1) //!< Number of ternary mutation rates

/**
 * @brief Self-adaptation method for mutating ternary conditions.
 */
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL };

/**
 * @brief Randomises a ternary condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
static void
cond_ternary_rand(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct CondTernary *cond = c->cond;
    for (int i = 0; i < cond->length; ++i) {
        if (rand_uniform(0, 1) < xcsf->cond->p_dontcare) {
            cond->string[i] = DONT_CARE;
        } else if (rand_uniform(0, 1) < 0.5) {
            cond->string[i] = '0';
        } else {
            cond->string[i] = '1';
        }
    }
}

/**
 * @brief Creates and initialises a ternary bitstring condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
void
cond_ternary_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondTernary *new = malloc(sizeof(struct CondTernary));
    new->length = xcsf->x_dim * xcsf->cond->bits;
    new->string = malloc(sizeof(char) * new->length);
    new->tmp_input = malloc(sizeof(char) * xcsf->cond->bits);
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
    cond_ternary_rand(xcsf, c);
}

/**
 * @brief Frees the memory used by a ternary condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be freed.
 */
void
cond_ternary_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondTernary *cond = c->cond;
    free(cond->string);
    free(cond->tmp_input);
    free(cond->mu);
    free(c->cond);
}

/**
 * @brief Copies a ternary condition from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
cond_ternary_copy(const struct XCSF *xcsf, struct Cl *dest,
                  const struct Cl *src)
{
    struct CondTernary *new = malloc(sizeof(struct CondTernary));
    const struct CondTernary *src_cond = src->cond;
    new->length = src_cond->length;
    new->string = malloc(sizeof(char) * src_cond->length);
    new->tmp_input = malloc(sizeof(char) * xcsf->cond->bits);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->string, src_cond->string, sizeof(char) * src_cond->length);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

/**
 * @brief Generates a ternary condition that matches the current input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is being covered.
 * @param [in] x The input state to cover.
 */
void
cond_ternary_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondTernary *cond = c->cond;
    const int bits = xcsf->cond->bits;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        float_to_binary(x[i], cond->tmp_input, bits);
        for (int j = 0; j < bits; ++j) {
            if (rand_uniform(0, 1) < xcsf->cond->p_dontcare) {
                cond->string[i * bits + j] = DONT_CARE;
            } else {
                cond->string[i * bits + j] = cond->tmp_input[j];
            }
        }
    }
}

/**
 * @brief Dummy update function.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_ternary_update(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x, const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Calculates whether a ternary condition matches an input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition to match.
 * @param [in] x The input state.
 * @return Whether the condition matches the input.
 */
bool
cond_ternary_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondTernary *cond = c->cond;
    const int bits = xcsf->cond->bits;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        float_to_binary(x[i], cond->tmp_input, bits);
        for (int j = 0; j < bits; ++j) {
            const char s = cond->string[i * bits + j];
            if (s != DONT_CARE && s != cond->tmp_input[j]) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Performs uniform crossover with two ternary conditions.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose condition is being crossed.
 * @param [in] c2 The second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
bool
cond_ternary_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2)
{
    const struct CondTernary *cond1 = c1->cond;
    const struct CondTernary *cond2 = c2->cond;
    bool changed = false;
    if (rand_uniform(0, 1) < xcsf->ea->p_crossover) {
        for (int i = 0; i < cond1->length; ++i) {
            if (rand_uniform(0, 1) < 0.5) {
                const char tmp = cond1->string[i];
                cond1->string[i] = cond2->string[i];
                cond2->string[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Mutates a ternary condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_ternary_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondTernary *cond = c->cond;
    sam_adapt(cond->mu, N_MU, MU_TYPE);
    bool changed = false;
    for (int i = 0; i < cond->length; ++i) {
        if (rand_uniform(0, 1) < cond->mu[0]) {
            if (cond->string[i] == DONT_CARE) {
                cond->string[i] = (rand_uniform(0, 1) < 0.5) ? '0' : '1';
            } else {
                cond->string[i] = DONT_CARE;
            }
            changed = true;
        }
    }
    return changed;
}

/**
 * @brief Returns whether classifier c1 has a condition more general than c2.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier whose condition is tested to be more general.
 * @param [in] c2 The classifier whose condition is tested to be more specific.
 * @return Whether the condition of c1 is more general than c2.
 */
bool
cond_ternary_general(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2)
{
    (void) xcsf;
    const struct CondTernary *cond1 = c1->cond;
    const struct CondTernary *cond2 = c2->cond;
    bool general = false;
    for (int i = 0; i < cond1->length; ++i) {
        if (cond1->string[i] != DONT_CARE &&
            cond1->string[i] != cond2->string[i]) {
            return false;
        }
        if (cond1->string[i] != cond2->string[i]) {
            general = true;
        }
    }
    return general;
}

/**
 * @brief Prints a ternary condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be printed.
 */
void
cond_ternary_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_ternary_json_export(xcsf, c));
}

/**
 * @brief Returns the size of a ternary condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition size to return.
 * @return The size of the condition.
 */
double
cond_ternary_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondTernary *cond = c->cond;
    return cond->length;
}

/**
 * @brief Writes a ternary condition to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_ternary_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    const struct CondTernary *cond = c->cond;
    s += fwrite(&cond->length, sizeof(int), 1, fp);
    s += fwrite(cond->string, sizeof(char), cond->length, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a ternary condition from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_ternary_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    size_t s = 0;
    struct CondTernary *new = malloc(sizeof(struct CondTernary));
    new->length = 0;
    s += fread(&new->length, sizeof(int), 1, fp);
    if (new->length < 1) {
        printf("cond_ternary_load(): read error\n");
        new->length = 1;
        exit(EXIT_FAILURE);
    }
    new->string = malloc(sizeof(char) * new->length);
    s += fread(new->string, sizeof(char), new->length, fp);
    new->tmp_input = malloc(sizeof(char) * xcsf->cond->bits);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a ternary condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_ternary_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondTernary *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "ternary");
    char buff[cond->length + 1];
    memcpy(buff, cond->string, sizeof(char) * cond->length);
    buff[cond->length] = '\0';
    cJSON_AddStringToObject(json, "string", buff);
    cJSON *mutation = cJSON_CreateDoubleArray(cond->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
