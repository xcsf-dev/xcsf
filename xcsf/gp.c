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
 * @date 2016--2020.
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

/**
 * @brief Self-adaptation method for mutating GP trees.
 */
static const int MU_TYPE[N_MU] = { SAM_RATE_SELECT };

/**
 * @brief Traverses a GP tree.
 * @param [in] tree The tree to traverse.
 * @param [in] p The position from which to traverse.
 * @return The position after traversal.
 */
static int
tree_traverse(int *tree, int p)
{
    if (tree[p] >= GP_NUM_FUNC) {
        ++p;
        return p;
    }
    switch (tree[p]) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
            ++p;
            return tree_traverse(tree, tree_traverse(tree, p));
        default:
            printf("tree_traverse() invalid function: %d\n", tree[p]);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Grows a random GP tree of specified max length and depth.
 * @details Only used to create an initial tree.
 * @param [in] args Tree GP parameters.
 * @param [in,out] buffer Buffer to hold the flattened GP tree generated.
 * @param [in] p The position from which to traverse (start at 0).
 * @param [in] max Maximum tree length.
 * @param [in] depth Maximum tree depth.
 * @return The position after traversal (i.e., tree length).
 */
static int
tree_grow(const struct GPTreeArgs *args, int *buffer, const int p,
          const int max, const int depth)
{
    int prim = rand_uniform_int(0, 2);
    if (p >= max) {
        return -1;
    }
    if (p == 0) {
        prim = 1;
    }
    if (prim == 0 || depth == 0) {
        const int max_term = GP_NUM_FUNC + args->n_constants + args->n_inputs;
        prim = rand_uniform_int(GP_NUM_FUNC, max_term);
        buffer[p] = prim;
        return p + 1;
    }
    prim = rand_uniform_int(0, GP_NUM_FUNC);
    switch (prim) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
            buffer[p] = prim;
            const int child = tree_grow(args, buffer, p + 1, max, depth - 1);
            if (child < 0) {
                return -1;
            }
            return tree_grow(args, buffer, child, max, depth - 1);
        default:
            printf("tree_grow() invalid function: %d\n", prim);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Creates a random GP tree.
 * @param [in] gp The GP tree being randomised.
 * @param [in] args Tree GP parameters.
 */
void
tree_rand(struct GPTree *gp, const struct GPTreeArgs *args)
{
    int buffer[args->max_len];
    gp->len = 0;
    do {
        gp->len = tree_grow(args, buffer, 0, args->max_len, args->init_depth);
    } while (gp->len < 0);
    gp->tree = malloc(sizeof(int) * gp->len);
    memcpy(gp->tree, buffer, sizeof(int) * gp->len);
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
tree_eval(struct GPTree *gp, const struct GPTreeArgs *args, const double *x)
{
    const int node = gp->tree[gp->p];
    ++(gp->p);
    if (node >= GP_NUM_FUNC + args->n_constants) {
        return x[node - GP_NUM_FUNC - args->n_constants];
    }
    if (node >= GP_NUM_FUNC) {
        return args->constants[node - GP_NUM_FUNC];
    }
    switch (node) {
        case ADD:
            return tree_eval(gp, args, x) + tree_eval(gp, args, x);
        case SUB:
            return tree_eval(gp, args, x) - tree_eval(gp, args, x);
        case MUL:
            return tree_eval(gp, args, x) * tree_eval(gp, args, x);
        case DIV: {
            const double num = tree_eval(gp, args, x);
            const double den = tree_eval(gp, args, x);
            if (den != 0) {
                return num / den;
            }
            return num;
        }
        default:
            printf("eval() invalid function: %d\n", node);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Prints a GP tree.
 * @param [in] gp The GP tree to print.
 * @param [in] args Tree GP parameters.
 * @param [in] p The position from which to traverse (start at 0).
 * @return The position after traversal.
 */
int
tree_print(const struct GPTree *gp, const struct GPTreeArgs *args, int p)
{
    const int node = gp->tree[p];
    if (node >= GP_NUM_FUNC) {
        if (node >= GP_NUM_FUNC + args->n_constants) {
            printf("IN:%d ", node - GP_NUM_FUNC - args->n_constants);
        } else {
            printf("%f", args->constants[node - GP_NUM_FUNC]);
        }
        ++p;
        return p;
    }
    printf("(");
    ++p;
    const int a1 = tree_print(gp, args, p);
    switch (node) {
        case ADD:
            printf(" + ");
            break;
        case SUB:
            printf(" - ");
            break;
        case MUL:
            printf(" * ");
            break;
        case DIV:
            printf(" / ");
            break;
        default:
            printf("tree_print() invalid function: %d\n", node);
            exit(EXIT_FAILURE);
    }
    const int a2 = tree_print(gp, args, a1);
    printf(")");
    return a2;
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
    dest->p = src->p;
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
tree_mutate(struct GPTree *gp, const struct GPTreeArgs *args)
{
    bool changed = false;
    sam_adapt(gp->mu, N_MU, MU_TYPE);
    const int max_term = GP_NUM_FUNC + args->n_constants + args->n_inputs;
    for (int i = 0; i < gp->len; ++i) {
        if (rand_uniform(0, 1) < gp->mu[0]) {
            changed = true;
            if (gp->tree[i] >= GP_NUM_FUNC) {
                gp->tree[i] = rand_uniform_int(GP_NUM_FUNC, max_term);
            } else {
                gp->tree[i] = rand_uniform_int(0, GP_NUM_FUNC);
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
    s += fwrite(&gp->p, sizeof(int), 1, fp);
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
    s += fread(&gp->p, sizeof(int), 1, fp);
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
tree_args_init(struct GPTreeArgs *args)
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
tree_args_free(struct GPTreeArgs *args)
{
    free(args->constants);
}

/**
 * @brief Prints Tree GP parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 */
void
tree_args_print(const struct GPTreeArgs *args)
{
    printf("n_inputs=%d", args->n_inputs);
    printf(", min_constant=%f", args->min);
    printf(", max_constant=%f", args->max);
    printf(", n_constants=%d", args->n_constants);
    printf(", init_depth=%d", args->init_depth);
    printf(", max_len=%d", args->max_len);
}

/**
 * @brief Saves Tree GP parameters.
 * @param [in] args Parameters for initialising and operating GP trees.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
tree_args_save(const struct GPTreeArgs *args, FILE *fp)
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
tree_args_load(struct GPTreeArgs *args, FILE *fp)
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
tree_args_init_constants(struct GPTreeArgs *args)
{
    if (args->constants != NULL) {
        free(args->constants);
    }
    args->constants = malloc(sizeof(double) * args->n_constants);
    for (int i = 0; i < args->n_constants; ++i) {
        args->constants[i] = rand_uniform(args->min, args->max);
    }
}
