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
 * @file ea.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2023.
 * @brief Evolutionary algorithm functions.
 */

#include "ea.h"
#include "cl.h"
#include "clset.h"
#include "utils.h"

/**
 * @brief Initialises offspring error and fitness.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1p First parent classifier.
 * @param [in] c2p Second parent classifier.
 * @param [in] c1 The first offspring classifier to initialise.
 * @param [in] c2 The second offspring classifier to initialise.
 * @param [in] cmod Whether crossover modified the offspring.
 */
static void
ea_init_offspring(const struct XCSF *xcsf, const struct Cl *c1p,
                  const struct Cl *c2p, struct Cl *c1, struct Cl *c2,
                  const bool cmod)
{
    if (cmod) {
        c1->err = xcsf->ea->err_reduc * ((c1p->err + c2p->err) * 0.5);
        c2->err = c1->err;
        c1->fit = c1p->fit / c1p->num;
        c2->fit = c2p->fit / c2p->num;
        c1->fit = xcsf->ea->fit_reduc * ((c1->fit + c2->fit) * 0.5);
        c2->fit = c1->fit;
    } else {
        c1->err = xcsf->ea->err_reduc * c1p->err;
        c2->err = xcsf->ea->err_reduc * c2p->err;
        c1->fit = xcsf->ea->fit_reduc * (c1p->fit / c1p->num);
        c2->fit = xcsf->ea->fit_reduc * (c2p->fit / c2p->num);
    }
}

/**
 * @brief Performs evolutionary algorithm subsumption.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The offspring classifier to attempt to subsume.
 * @param [in] c1p First parent classifier.
 * @param [in] c2p Second parent classifier.
 * @param [in] set The set in which the EA is being run.
 */
static void
ea_subsume(struct XCSF *xcsf, struct Cl *c, struct Cl *c1p, struct Cl *c2p,
           const struct Set *set)
{
    // check if either parent subsumes the offspring
    if (cl_subsumer(xcsf, c1p) && cl_general(xcsf, c1p, c)) {
        ++(c1p->num);
        ++(xcsf->pset.num);
        cl_free(xcsf, c);
    } else if (cl_subsumer(xcsf, c2p) && cl_general(xcsf, c2p, c)) {
        ++(c2p->num);
        ++(xcsf->pset.num);
        cl_free(xcsf, c);
    }
    // attempt to find a random subsumer from the set
    else {
        struct Clist *candidates[set->size];
        int choices = 0;
        for (struct Clist *iter = set->list; iter != NULL; iter = iter->next) {
            if (cl_subsumer(xcsf, iter->cl) && cl_general(xcsf, iter->cl, c)) {
                candidates[choices] = iter;
                ++choices;
            }
        }
        if (choices > 0) { // found
            ++(candidates[rand_uniform_int(0, choices)]->cl->num);
            ++(xcsf->pset.num);
            cl_free(xcsf, c);
        }
        // if no subsumers are found the offspring is added to the population
        else {
            clset_add(&xcsf->pset, c);
        }
    }
}

/**
 * @brief Adds offspring to the population.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set in which the EA is being run.
 * @param [in] c1p First parent classifier.
 * @param [in] c2p Second parent classifier.
 * @param [in] c1 The offspring classifier to add.
 * @param [in] cmod Whether crossover modified the offspring.
 * @param [in] mmod Whether mutation modified the offspring.
 */
static void
ea_add(struct XCSF *xcsf, const struct Set *set, struct Cl *c1p, struct Cl *c2p,
       struct Cl *c1, const bool cmod, const bool mmod)
{
    if (!cmod && !mmod) {
        ++(c1p->num);
        ++(xcsf->pset.num);
        cl_free(xcsf, c1);
    } else if (xcsf->ea->subsumption) {
        ea_subsume(xcsf, c1, c1p, c2p, set);
    } else {
        clset_add(&xcsf->pset, c1);
    }
}

/**
 * @brief Selects a classifier from the set via roulette wheel.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to select from.
 * @param [in] fit_sum The sum of all the fitnesses in the set.
 * @return A pointer to the selected classifier.
 */
static struct Cl *
ea_select_rw(const struct XCSF *xcsf, const struct Set *set,
             const double fit_sum)
{
    (void) xcsf;
    const double p = rand_uniform(0, fit_sum);
    const struct Clist *iter = set->list;
    double sum = iter->cl->fit;
    while (p > sum) {
        iter = iter->next;
        sum += iter->cl->fit;
    }
    return iter->cl;
}

/**
 * @brief Selects a classifier from the set via tournament.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to select from.
 * @return A pointer to the selected classifier.
 */
static struct Cl *
ea_select_tournament(const struct XCSF *xcsf, const struct Set *set)
{
    struct Cl *winner = NULL;
    while (winner == NULL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            if ((rand_uniform(0, 1) < xcsf->ea->select_size) &&
                (winner == NULL || iter->cl->fit > winner->fit)) {
                winner = iter->cl;
            }
            iter = iter->next;
        }
    }
    return winner;
}

/**
 * @brief Selects two parents.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set in which the EA is being run.
 * @param [out] c1p First selected parent classifier.
 * @param [out] c2p Second selected parent classifier.
 */
static void
ea_select(const struct XCSF *xcsf, const struct Set *set, struct Cl **c1p,
          struct Cl **c2p)
{
    if (xcsf->ea->select_type == EA_SELECT_ROULETTE) {
        const double fit_sum = clset_total_fit(set);
        *c1p = ea_select_rw(xcsf, set, fit_sum);
        *c2p = ea_select_rw(xcsf, set, fit_sum);
    } else {
        *c1p = ea_select_tournament(xcsf, set);
        *c2p = ea_select_tournament(xcsf, set);
    }
}

/**
 * @brief Executes the evolutionary algorithm (EA).
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set in which to run the EA.
 */
void
ea(struct XCSF *xcsf, const struct Set *set)
{
    ++(xcsf->time);
    if (set->size == 0 || xcsf->time - clset_mean_time(set) < xcsf->ea->theta) {
        return; // not yet time to run the EA
    }
    clset_set_times(xcsf, set);
    // select parents
    struct Cl *c1p = NULL;
    struct Cl *c2p = NULL;
    ea_select(xcsf, set, &c1p, &c2p);
    // create offspring
    for (int i = 0; i * 2 < xcsf->ea->lambda; ++i) {
        // create copies of parents
        struct Cl *c1 = malloc(sizeof(struct Cl));
        struct Cl *c2 = malloc(sizeof(struct Cl));
        cl_init(xcsf, c1, c1p->size, c1p->time);
        cl_init(xcsf, c2, c2p->size, c2p->time);
        cl_copy(xcsf, c1, c1p);
        cl_copy(xcsf, c2, c2p);
        // apply evolutionary operators to offspring
        const bool cmod = cl_crossover(xcsf, c1, c2);
        const bool m1mod = cl_mutate(xcsf, c1);
        const bool m2mod = cl_mutate(xcsf, c2);
        // initialise parameters
        ea_init_offspring(xcsf, c1p, c2p, c1, c2, cmod);
        // add to population
        ea_add(xcsf, set, c1p, c2p, c1, cmod, m1mod);
        ea_add(xcsf, set, c2p, c1p, c2, cmod, m2mod);
    }
    clset_pset_enforce_limit(xcsf);
}

/**
 * @brief Initialises default evolutionary algorithm parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
ea_param_defaults(struct XCSF *xcsf)
{
    ea_param_set_select_type(xcsf, EA_SELECT_ROULETTE);
    ea_param_set_select_size(xcsf, 0.4);
    ea_param_set_theta(xcsf, 50);
    ea_param_set_lambda(xcsf, 2);
    ea_param_set_p_crossover(xcsf, 0.8);
    ea_param_set_subsumption(xcsf, false);
    ea_param_set_err_reduc(xcsf, 1);
    ea_param_set_fit_reduc(xcsf, 0.1);
    ea_param_set_pred_reset(xcsf, false);
}

/**
 * @brief Returns a json formatted string representation of the EA parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
char *
ea_param_json_export(const struct XCSF *xcsf)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "select_type",
                            ea_type_as_string(xcsf->ea->select_type));
    if (xcsf->ea->select_type == EA_SELECT_TOURNAMENT) {
        cJSON_AddNumberToObject(json, "select_size", xcsf->ea->select_size);
    }
    cJSON_AddNumberToObject(json, "theta_ea", xcsf->ea->theta);
    cJSON_AddNumberToObject(json, "lambda", xcsf->ea->lambda);
    cJSON_AddNumberToObject(json, "p_crossover", xcsf->ea->p_crossover);
    cJSON_AddNumberToObject(json, "err_reduc", xcsf->ea->err_reduc);
    cJSON_AddNumberToObject(json, "fit_reduc", xcsf->ea->fit_reduc);
    cJSON_AddBoolToObject(json, "subsumption", xcsf->ea->subsumption);
    cJSON_AddBoolToObject(json, "pred_reset", xcsf->ea->pred_reset);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Sets the EA parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 */
void
ea_param_json_import(struct XCSF *xcsf, cJSON *json)
{
    for (cJSON *iter = json; iter != NULL; iter = iter->next) {
        if (strncmp(iter->string, "select_type\0", 12) == 0 &&
            cJSON_IsString(iter)) {
            if (ea_param_set_type_string(xcsf, iter->valuestring) ==
                EA_SELECT_INVALID) {
                printf("Invalid EA SELECT_TYPE: %s\n", iter->valuestring);
                printf("Options: {%s}\n", EA_SELECT_OPTIONS);
                exit(EXIT_FAILURE);
            }
        } else if (strncmp(iter->string, "select_size\0", 12) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_select_size(xcsf, iter->valuedouble));
        } else if (strncmp(iter->string, "theta_ea\0", 9) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_theta(xcsf, iter->valuedouble));
        } else if (strncmp(iter->string, "lambda\0", 7) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_lambda(xcsf, iter->valueint));
        } else if (strncmp(iter->string, "p_crossover\0", 12) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_p_crossover(xcsf, iter->valuedouble));
        } else if (strncmp(iter->string, "err_reduc\0", 10) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_err_reduc(xcsf, iter->valuedouble));
        } else if (strncmp(iter->string, "fit_reduc\0", 10) == 0 &&
                   cJSON_IsNumber(iter)) {
            catch_error(ea_param_set_fit_reduc(xcsf, iter->valuedouble));
        } else if (strncmp(iter->string, "subsumption\0", 12) == 0 &&
                   cJSON_IsBool(iter)) {
            const bool sub = true ? iter->type == cJSON_True : false;
            catch_error(ea_param_set_subsumption(xcsf, sub));
        } else if (strncmp(iter->string, "pred_reset\0", 11) == 0 &&
                   cJSON_IsBool(iter)) {
            const bool reset = true ? iter->type == cJSON_True : false;
            catch_error(ea_param_set_pred_reset(xcsf, reset));
        } else {
            printf("Error importing EA parameter %s\n", iter->string);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Saves evolutionary algorithm parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
ea_param_save(const struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->ea->select_type, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ea->select_size, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ea->theta, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ea->lambda, sizeof(int), 1, fp);
    s += fwrite(&xcsf->ea->p_crossover, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ea->err_reduc, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ea->fit_reduc, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ea->subsumption, sizeof(bool), 1, fp);
    s += fwrite(&xcsf->ea->pred_reset, sizeof(bool), 1, fp);
    return s;
}

/**
 * @brief Loads evolutionary algorithm parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
ea_param_load(struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fread(&xcsf->ea->select_type, sizeof(int), 1, fp);
    s += fread(&xcsf->ea->select_size, sizeof(double), 1, fp);
    s += fread(&xcsf->ea->theta, sizeof(double), 1, fp);
    s += fread(&xcsf->ea->lambda, sizeof(int), 1, fp);
    s += fread(&xcsf->ea->p_crossover, sizeof(double), 1, fp);
    s += fread(&xcsf->ea->err_reduc, sizeof(double), 1, fp);
    s += fread(&xcsf->ea->fit_reduc, sizeof(double), 1, fp);
    s += fread(&xcsf->ea->subsumption, sizeof(bool), 1, fp);
    s += fread(&xcsf->ea->pred_reset, sizeof(bool), 1, fp);
    return s;
}

/**
 * @brief Returns a string representation of an EA select type from an integer.
 * @param [in] type Integer representation of an EA select type.
 * @return String representing the name of the EA select type.
 */
const char *
ea_type_as_string(const int type)
{
    if (type == EA_SELECT_ROULETTE) {
        return EA_STRING_ROULETTE;
    }
    if (type == EA_SELECT_TOURNAMENT) {
        return EA_STRING_TOURNAMENT;
    }
    printf("ea_type_as_string(): invalid type: %d\n", type);
    exit(EXIT_FAILURE);
}

/**
 * @brief Returns the integer representation of an EA selection type.
 * @param [in] type String representation of an EA type.
 * @return Integer representing the EA type.
 */
int
ea_type_as_int(const char *type)
{
    if (strncmp(type, EA_STRING_ROULETTE, 9) == 0) {
        return EA_SELECT_ROULETTE;
    }
    if (strncmp(type, EA_STRING_TOURNAMENT, 11) == 0) {
        return EA_SELECT_TOURNAMENT;
    }
    return EA_SELECT_INVALID;
}

/* parameter setters */

const char *
ea_param_set_select_size(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid EA SELECT_SIZE. Range: [0,1]";
    }
    xcsf->ea->select_size = a;
    return NULL;
}

const char *
ea_param_set_theta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "EA THETA must be >= 0";
    }
    xcsf->ea->theta = a;
    return NULL;
}

const char *
ea_param_set_p_crossover(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid EA P_CROSSOVER. Range: [0,1]";
    }
    xcsf->ea->p_crossover = a;
    return NULL;
}

const char *
ea_param_set_lambda(struct XCSF *xcsf, const int a)
{
    if (a < 2) {
        return "EA LAMBDA must be >= 2";
    }
    xcsf->ea->lambda = a;
    return NULL;
}

const char *
ea_param_set_err_reduc(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid EA ERR_REDUC. Range: [0,1]";
    }
    xcsf->ea->err_reduc = a;
    return NULL;
}

const char *
ea_param_set_fit_reduc(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid EA FIT_REDUC. Range: [0,1]";
    }
    xcsf->ea->fit_reduc = a;
    return NULL;
}

const char *
ea_param_set_subsumption(struct XCSF *xcsf, const bool a)
{
    xcsf->ea->subsumption = a;
    return NULL;
}

const char *
ea_param_set_pred_reset(struct XCSF *xcsf, const bool a)
{
    xcsf->ea->pred_reset = a;
    return NULL;
}

int
ea_param_set_select_type(struct XCSF *xcsf, const int a)
{
    if (a == EA_SELECT_ROULETTE || a == EA_SELECT_TOURNAMENT) {
        xcsf->ea->select_type = a;
        return a;
    }
    return EA_SELECT_INVALID;
}

int
ea_param_set_type_string(struct XCSF *xcsf, const char *a)
{
    const int type = ea_type_as_int(a);
    if (type != EA_SELECT_INVALID) {
        xcsf->ea->select_type = type;
        return type;
    }
    return EA_SELECT_INVALID;
}
