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
 * @file gp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of GP trees based upon TinyGP.
 * @see Poli, Langdon, and McPhee (2008) "A Field Guide to Genetic Programming"
 */

#include "gp.h"
#include "sam.h"
#include "utils.h"

#define GP_NUM_FUNC (4) //!< Number of selectable GP functions
#define ADD (0) //!< Addition function
#define SUB (1) //!< Subtraction function
#define MUL (2) //!< Multiplication function
#define DIV (3) //!< Division function

#define N_MU (1) //!< Number of tree-GP mutation rates
#define RET_MIN (-1000) //!< Minimum tree return value
#define RET_MAX (1000) //!< Maximum tree return value

/**
 * @brief Self-adaptation method for mutating GP trees.
 */
static const int MU_TYPE[N_MU] = { SAM_RATE_SELECT };

/**
 * @brief Traverses a GP tree.
 * @param [in] tree The tree to traverse.
 * @param [in] pos The position from which to traverse.
 * @return The position after traversal.
 */
static int
tree_traverse(int *tree, int pos)
{
    if (tree[pos] >= GP_NUM_FUNC) {
        ++pos;
        return pos;
    }
    ++pos;
    return tree_traverse(tree, tree_traverse(tree, pos));
}

/**
 * @brief Grows a random GP tree of specified max length and depth.
 * @details Only used to create an initial tree.
 * @param [in] args Tree GP parameters.
 * @param [in,out] tree Vector holding the flattened GP tree generated.
 * @param [in] pos The position from which to traverse (start at 0).
 * @param [in] max Maximum tree length.
 * @param [in] depth Maximum tree depth.
 * @return The position after traversal (i.e., tree length).
 */
static int
tree_grow(const struct ArgsGPTree *args, int *tree, const int pos,
          const int max, const int depth)
{
    if (pos >= max) {
        return -1;
    }
    if (depth == 0 || (pos != 0 && rand_uniform(0, 1) < 0.5)) {
        const int max_term = GP_NUM_FUNC + args->n_constants + args->n_inputs;
        tree[pos] = rand_uniform_int(GP_NUM_FUNC, max_term);
        return pos + 1;
    }
    tree[pos] = rand_uniform_int(0, GP_NUM_FUNC);
    const int child = tree_grow(args, tree, pos + 1, max, depth - 1);
    if (child < 0) {
        return -1;
    }
    return tree_grow(args, tree, child, max, depth - 1);
}

/**
 * @brief Creates a random GP tree.
 * @param [in] gp The GP tree being randomised.
 * @param [in] args Tree GP parameters.
 */
void
tree_rand(struct GPTree *gp, const struct ArgsGPTree *args)
{
    gp->tree = malloc(sizeof(int) * args->max_len);
    gp->len = 0;
    while (gp->len < 1) {
        gp->len = tree_grow(args, gp->tree, 0, args->max_len, args->init_depth);
    }
    gp->tree = realloc(gp->tree, sizeof(int) * gp->len);
    gp->mu = malloc(sizeof(double) * N_MU);
    sam_init(gp->mu, N_MU, MU_TYPE);
}

/**
 * @brief Frees a GP tree.
 * @param [in] gp The GP tree to free.
 */
void
tree_free(const struct GPTree *gp)
{
    free(gp->tree);
    free(gp->mu);
}

/**
 * @brief Evaluates a GP tree.
 * @param [in] gp The GP tree to evaluate.
 * @param [in] args Tree GP parameters.
 * @param [in] x The input state.
 * @return The result from evaluating the GP tree.
 */
double
tree_eval(struct GPTree *gp, const struct ArgsGPTree *args, const double *x)
{
    const int node = gp->tree[gp->pos];
    ++(gp->pos);
    if (node >= GP_NUM_FUNC + args->n_constants) {
        return x[node - GP_NUM_FUNC - args->n_constants];
    }
    if (node >= GP_NUM_FUNC) {
        return args->constants[node - GP_NUM_FUNC];
    }
    const double a = clamp(tree_eval(gp, args, x), RET_MIN, RET_MAX);
    const double b = clamp(tree_eval(gp, args, x), RET_MIN, RET_MAX);
    switch (node) {
        case ADD:
            return a + b;
        case SUB:
            return a - b;
        case MUL:
            return a * b;
        case DIV:
            return (b != 0) ? (a / b) : a;
        default:
            printf("tree_eval() invalid function: %d\n", node);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a json formatted string represetation of a GP tree.
 * @param [in] gp The GP tree to print.
 * @param [in] args Tree GP parameters.
 * @param [in] pos The position from which to traverse (start at 0).
 * @param [in] json cJSON object.
 * @return The position after traversal.
 */
static int
tree_string(const struct GPTree *gp, const struct ArgsGPTree *args, int pos,
            cJSON *json)
{
    const int node = gp->tree[pos];
    if (node >= GP_NUM_FUNC) {
        if (node >= GP_NUM_FUNC + args->n_constants) {
            const int val = node - GP_NUM_FUNC - args->n_constants;
            char buff[256];
            snprintf(buff, 256, "feature_%d", val);
            cJSON *input = cJSON_CreateString(buff);
            cJSON_AddItemToArray(json, input);
        } else {
            const double val = args->constants[node - GP_NUM_FUNC];
            cJSON *constant = cJSON_CreateNumber(val);
            cJSON_AddItemToArray(json, constant);
        }
        ++pos;
        return pos;
    }
    cJSON *leftp = cJSON_CreateString("(");
    cJSON_AddItemToArray(json, leftp);
    ++pos;
    const int a1 = tree_string(gp, args, pos, json);
    cJSON *func;
    switch (node) {
        case ADD:
            func = cJSON_CreateString("+");
            break;
        case SUB:
            func = cJSON_CreateString("-");
            break;
        case MUL:
            func = cJSON_CreateString("*");
            break;
        case DIV:
            func = cJSON_CreateString("/");
            break;
        default:
            printf("tree_string() invalid function: %d\n", node);
            exit(EXIT_FAILURE);
    }
    cJSON_AddItemToArray(json, func);
    const int a2 = tree_string(gp, args, a1, json);
    cJSON *rightp = cJSON_CreateString(")");
    cJSON_AddItemToArray(json, rightp);
    return a2;
}

/**
 * @brief Returns a json formatted string representation of a GP tree.
 * @param [in] gp The GP tree to return.
 * @param [in] args Tree GP parameters.
 * @return String encoded in json format.
 */
const char *
tree_json_export(const struct GPTree *gp, const struct ArgsGPTree *args)
{
    cJSON *json = cJSON_CreateObject();
    cJSON *tree = cJSON_CreateArray();
    cJSON_AddItemToObject(json, "array", tree);
    tree_string(gp, args, 0, tree);
    cJSON *mutation = cJSON_CreateDoubleArray(gp->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Prints a GP tree.
 * @param [in] gp The GP tree to print.
 * @param [in] args Tree GP parameters.
 */
void
tree_print(const struct GPTree *gp, const struct ArgsGPTree *args)
{
    printf("%s\n", tree_json_export(gp, args));
}

/**
 * @brief Copies a GP tree.
 * @param [in] dest The destination GP tree.
 * @param [in] src The source GP tree.
 */
void
tree_copy(struct GPTree *dest, const struct GPTree *src)
{
    dest->len = src->len;
    dest->tree = malloc(sizeof(int) * src->len);
    memcpy(dest->tree, src->tree, sizeof(int) * src->len);
    dest->pos = src->pos;
    dest->mu = malloc(sizeof(double) * N_MU);
    memcpy(dest->mu, src->mu, sizeof(double) * N_MU);
}

/**
 * @brief Performs sub-tree crossover.
 * @param [in] p1 The first GP tree to perform crossover.
 * @param [in] p2 The second GP tree to perform crossover.
 */
void
tree_crossover(struct GPTree *p1, struct GPTree *p2)
{
    const int len1 = p1->len;
    const int len2 = p2->len;
    const int start1 = rand_uniform_int(0, len1);
    const int end1 = tree_traverse(p1->tree, start1);
    const int start2 = rand_uniform_int(0, len2);
    const int end2 = tree_traverse(p2->tree, start2);
    const int nlen1 = start1 + (end2 - start2) + (len1 - end1);
    int *new1 = malloc(sizeof(int) * nlen1);
    memcpy(&new1[0], &p1->tree[0], sizeof(int) * start1);
    memcpy(&new1[start1], &p2->tree[start2], sizeof(int) * (end2 - start2));
    memcpy(&new1[start1 + (end2 - start2)], &p1->tree[end1],
           sizeof(int) * (len1 - end1));
    const int nlen2 = start2 + (end1 - start1) + (len2 - end2);
    int *new2 = malloc(sizeof(int) * nlen2);
    memcpy(&new2[0], &p2->tree[0], sizeof(int) * start2);
    memcpy(&new2[start2], &p1->tree[start1], sizeof(int) * (end1 - start1));
    memcpy(&new2[start2 + (end1 - start1)], &p2->tree[end2],
           sizeof(int) * (len2 - end2));
    free(p1->tree);
    free(p2->tree);
    p1->tree = new1;
    p2->tree = new2;
    p1->len = tree_traverse(p1->tree, 0);
    p2->len = tree_traverse(p2->tree, 0);
}

/**
 * @brief Performs point mutation on a GP tree.
 * @details Terminals are randomly replaced with other terminals and functions
 * are randomly replaced with other functions.
 * @param [in] gp The GP tree to be mutated.
 * @param [in] args Tree GP parameters.
 * @return Whether any alterations were made.
 */
bool
tree_mutate(struct GPTree *gp, const struct ArgsGPTree *args)
{
    bool changed = false;
    sam_adapt(gp->mu, N_MU, MU_TYPE);
    const int max_term = GP_NUM_FUNC + args->n_constants + args->n_inputs;
    for (int i = 0; i < gp->len; ++i) {
        if (rand_uniform(0, 1) < gp->mu[0]) {
            const int orig = gp->tree[i];
            if (gp->tree[i] >= GP_NUM_FUNC) {
                gp->tree[i] = rand_uniform_int(GP_NUM_FUNC, max_term);
            } else {
                gp->tree[i] = rand_uniform_int(0, GP_NUM_FUNC);
            }
            if (gp->tree[i] != orig) {
                changed = true;
            }
        }
    }
    return changed;
}

/**
 * @brief Writes the GP tree to a file.
 * @param [in] gp The GP tree to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
tree_save(const struct GPTree *gp, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&gp->pos, sizeof(int), 1, fp);
    s += fwrite(&gp->len, sizeof(int), 1, fp);
    s += fwrite(gp->tree, sizeof(int), gp->len, fp);
    s += fwrite(gp->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a GP tree from a file.
 * @param [in] gp The GP tree to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
tree_load(struct GPTree *gp, FILE *fp)
{
    size_t s = 0;
    s += fread(&gp->pos, sizeof(int), 1, fp);
    s += fread(&gp->len, sizeof(int), 1, fp);
    if (gp->len < 1) {
        printf("tree_load(): read error\n");
        gp->len = 1;
        exit(EXIT_FAILURE);
    }
    gp->tree = malloc(sizeof(int) * gp->len);
    s += fread(gp->tree, sizeof(int), gp->len, fp);
    s += fread(gp->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Sets tree GP parameters to default values.
 * @param [in] args Parameters for initialising and operating GP trees.
 */
void
tree_args_init(struct ArgsGPTree *args)
{
    args->max = 0;
    args->min = 0;
    args->n_inputs = 0;
    args->n_constants = 0;
    args->init_depth = 0;
    args->max_len = 0;
    args->constants = NULL;
}

/**
 * @brief Frees memory used by GP tree parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 */
void
tree_args_free(struct ArgsGPTree *args)
{
    free(args->constants);
}

/**
 * @brief Returns a json formatted string of the GP tree parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 * @return String encoded in json format.
 */
const char *
tree_args_json_export(const struct ArgsGPTree *args)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "min_constant", args->min);
    cJSON_AddNumberToObject(json, "max_constant", args->max);
    cJSON_AddNumberToObject(json, "n_constants", args->n_constants);
    cJSON_AddNumberToObject(json, "init_depth", args->init_depth);
    cJSON_AddNumberToObject(json, "max_len", args->max_len);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Saves Tree GP parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
tree_args_save(const struct ArgsGPTree *args, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&args->max, sizeof(double), 1, fp);
    s += fwrite(&args->min, sizeof(double), 1, fp);
    s += fwrite(&args->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&args->n_constants, sizeof(int), 1, fp);
    s += fwrite(&args->init_depth, sizeof(int), 1, fp);
    s += fwrite(&args->max_len, sizeof(int), 1, fp);
    s += fwrite(args->constants, sizeof(double), args->n_constants, fp);
    return s;
}

/**
 * @brief Loads Tree GP parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements read.
 */
size_t
tree_args_load(struct ArgsGPTree *args, FILE *fp)
{
    size_t s = 0;
    s += fread(&args->max, sizeof(double), 1, fp);
    s += fread(&args->min, sizeof(double), 1, fp);
    s += fread(&args->n_inputs, sizeof(int), 1, fp);
    s += fread(&args->n_constants, sizeof(int), 1, fp);
    s += fread(&args->init_depth, sizeof(int), 1, fp);
    s += fread(&args->max_len, sizeof(int), 1, fp);
    s += fread(args->constants, sizeof(double), args->n_constants, fp);
    return s;
}

/**
 * @brief Builds global constants used by GP trees.
 * @param [in] args Parameters for initialising and operating GP trees.
 */
void
tree_args_init_constants(struct ArgsGPTree *args)
{
    if (args->constants != NULL) {
        free(args->constants);
    }
    args->constants = malloc(sizeof(double) * args->n_constants);
    for (int i = 0; i < args->n_constants; ++i) {
        args->constants[i] = rand_uniform(args->min, args->max);
    }
}

/* parameter setters */

void
tree_param_set_max(struct ArgsGPTree *args, const double a)
{
    args->max = a;
}

void
tree_param_set_min(struct ArgsGPTree *args, const double a)
{
    args->min = a;
}

void
tree_param_set_n_inputs(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP N_INPUTS too small\n");
        args->n_inputs = 1;
    } else {
        args->n_inputs = a;
    }
}

void
tree_param_set_n_constants(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP N_CONSTANTS too small\n");
        args->n_constants = 1;
    } else {
        args->n_constants = a;
    }
}

void
tree_param_set_init_depth(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP INIT_DEPTH too small\n");
        args->init_depth = 1;
    } else {
        args->init_depth = a;
    }
}

void
tree_param_set_max_len(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP MAX_LEN too small\n");
        args->max_len = 1;
    } else {
        args->max_len = a;
    }
}
