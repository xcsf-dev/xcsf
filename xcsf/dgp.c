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
 * @file dgp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of dynamical GP graphs with fuzzy activations.
 */

#include "dgp.h"
#include "sam.h"
#include "utils.h"

#define FUZZY_NOT (0) //!< Fuzzy NOT function
#define FUZZY_CFMQVS_AND (1) //!< Fuzzy AND (CFMQVS) function
#define FUZZY_CFMQVS_OR (2) //!< Fuzzy OR (CFMQVS) function
#define NUM_FUNC (3) //!< Number of selectable node functions
#define N_MU (3) //!< Number of DGP graph mutation rates

/**
 * @brief Self-adaptation method for mutating DGP graphs.
 */
static const int MU_TYPE[N_MU] = {
    SAM_LOG_NORMAL, //!< Node function mutation
    SAM_LOG_NORMAL, //!< Connectivity mutation
    SAM_UNIFORM //!< Number of update cycles mutation
};

/**
 * @brief Returns a random connection.
 * @param [in] n_nodes The number of nodes in the graph.
 * @param [in] n_inputs The number of external inputs to the graph.
 */
static int
random_connection(const int n_nodes, const int n_inputs)
{
    // another node within the graph
    if (rand_uniform(0, 1) < 0.5) {
        return rand_uniform_int(0, n_nodes) + n_inputs;
    }
    // external input
    return rand_uniform_int(0, n_inputs);
}

/**
 * @brief Mutates the node functions within a DGP graph.
 * @param [in] dgp The DGP graph to be mutated.
 * @return Whether any alterations were made.
 */
static bool
graph_mutate_functions(struct Graph *dgp)
{
    bool mod = false;
    for (int i = 0; i < dgp->n; ++i) {
        if (rand_uniform(0, 1) < dgp->mu[0]) {
            const int orig = dgp->function[i];
            dgp->function[i] = rand_uniform_int(0, NUM_FUNC);
            if (orig != dgp->function[i]) {
                mod = true;
            }
        }
    }
    return mod;
}

/**
 * @brief Mutates the connectivity of a DGP graph.
 * @param [in] dgp The DGP graph to be mutated.
 * @return Whether any alterations were made.
 */
static bool
graph_mutate_connectivity(struct Graph *dgp)
{
    bool mod = false;
    for (int i = 0; i < dgp->klen; ++i) {
        if (rand_uniform(0, 1) < dgp->mu[1]) {
            const int orig = dgp->connectivity[i];
            dgp->connectivity[i] = random_connection(dgp->n, dgp->n_inputs);
            if (orig != dgp->connectivity[i]) {
                mod = true;
            }
        }
    }
    return mod;
}

/**
 * @brief Mutates the number of update cycles performed by a DGP graph.
 * @param [in] dgp The DGP graph to be mutated.
 * @return Whether any alterations were made.
 */
static bool
graph_mutate_cycles(struct Graph *dgp)
{
    const int n = (int) round((2 * dgp->mu[2]) - 1);
    if (dgp->t + n < 1 || dgp->t + n > dgp->max_t) {
        return false;
    }
    dgp->t += n;
    return true;
}

/**
 * @brief Returns the result from applying a specified activation function.
 * @param [in] function The activation function to apply.
 * @param [in] inputs The input to the activation function.
 * @param [in] K The number of inputs to the activation function.
 * @return The result from applying the activation function.
 */
static double
node_activate(int function, const double *inputs, const int K)
{
    double state = 0;
    switch (function) {
        case FUZZY_NOT:
            state = 1 - inputs[0];
            break;
        case FUZZY_CFMQVS_AND:
            state = inputs[0];
            for (int i = 1; i < K; ++i) {
                state *= inputs[i];
            }
            break;
        case FUZZY_CFMQVS_OR:
            state = inputs[0];
            for (int i = 1; i < K; ++i) {
                state += inputs[i];
            }
            break;
        default: // Invalid function
            printf("Error updating node: Invalid function: %d\n", function);
            exit(EXIT_FAILURE);
    }
    state = clamp(state, 0, 1);
    return state;
}

/**
 * @brief Returns the name of a specified node function.
 * @param [in] function The node function.
 * @return The name of the node function.
 */
static const char *
function_string(const int function)
{
    switch (function) {
        case FUZZY_NOT:
            return "Fuzzy NOT";
        case FUZZY_CFMQVS_AND:
            return "Fuzzy AND";
        case FUZZY_CFMQVS_OR:
            return "Fuzzy OR";
        default:
            printf("function_string(): invalid node function: %d\n", function);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Performs a synchronous update.
 * @param [in] dgp The DGP graph to update.
 * @param [in] inputs The inputs to the graph.
 */
static void
synchronous_update(const struct Graph *dgp, const double *inputs)
{
    for (int i = 0; i < dgp->n; ++i) {
        for (int k = 0; k < dgp->max_k; ++k) {
            const int c = dgp->connectivity[i * dgp->max_k + k];
            if (c < dgp->n_inputs) { // external input
                dgp->tmp_input[k] = inputs[c];
            } else { // another node within the graph
                dgp->tmp_input[k] = dgp->state[c - dgp->n_inputs];
            }
        }
        dgp->tmp_state[i] =
            node_activate(dgp->function[i], dgp->tmp_input, dgp->max_k);
    }
    memcpy(dgp->state, dgp->tmp_state, sizeof(double) * dgp->n);
}

/**
 * @brief Initialises a new DGP graph.
 * @param [in] dgp The DGP graph to initialise.
 * @param [in] args Parameters for initialising a DGP graph.
 */
void
graph_init(struct Graph *dgp, const struct ArgsDGP *args)
{
    dgp->n_inputs = args->n_inputs;
    dgp->n = args->n;
    dgp->t = args->max_t;
    dgp->max_t = args->max_t;
    dgp->max_k = args->max_k;
    dgp->evolve_cycles = args->evolve_cycles;
    dgp->klen = dgp->n * dgp->max_k;
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->tmp_state = malloc(sizeof(double) * dgp->n);
    dgp->tmp_input = malloc(sizeof(double) * dgp->max_k);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
    dgp->mu = malloc(sizeof(double) * N_MU);
    sam_init(dgp->mu, N_MU, MU_TYPE);
}

/**
 * @brief Copies a DGP graph.
 * @param [in] dest The destination DGP graph.
 * @param [in] src The source DGP graph.
 */
void
graph_copy(struct Graph *dest, const struct Graph *src)
{
    dest->t = src->t;
    dest->n = src->n;
    dest->klen = src->klen;
    dest->max_k = src->max_k;
    dest->max_t = src->max_t;
    dest->n_inputs = src->n_inputs;
    dest->evolve_cycles = src->evolve_cycles;
    memcpy(dest->state, src->state, sizeof(double) * src->n);
    memcpy(dest->initial_state, src->initial_state, sizeof(double) * src->n);
    memcpy(dest->function, src->function, sizeof(int) * src->n);
    memcpy(dest->connectivity, src->connectivity, sizeof(int) * src->klen);
    memcpy(dest->mu, src->mu, sizeof(double) * N_MU);
}

/**
 * @brief Returns the current state of a specified node in the graph.
 * @param [in] dgp The DGP graph to output.
 * @param [in] IDX Which node within the graph to output.
 * @return The current state of the specified node.
 */
double
graph_output(const struct Graph *dgp, const int IDX)
{
    return dgp->state[IDX];
}

/**
 * @brief Resets the states to their initial state.
 * @param [in] dgp The DGP graph to reset.
 */
void
graph_reset(const struct Graph *dgp)
{
    for (int i = 0; i < dgp->n; ++i) {
        dgp->state[i] = dgp->initial_state[i];
    }
}

/**
 * @brief Randomises a specified DGP graph.
 * @param [in] dgp The DGP graph to randomise.
 */
void
graph_rand(struct Graph *dgp)
{
    if (dgp->evolve_cycles) {
        dgp->t = rand_uniform_int(1, dgp->max_t);
    }
    for (int i = 0; i < dgp->n; ++i) {
        dgp->function[i] = rand_uniform_int(0, NUM_FUNC);
        dgp->initial_state[i] = rand_uniform(0, 1);
        dgp->state[i] = rand_uniform(0, 1);
    }
    for (int i = 0; i < dgp->klen; ++i) {
        dgp->connectivity[i] = random_connection(dgp->n, dgp->n_inputs);
    }
}

/**
 * @brief Updates a DGP graph T cycles.
 * @param [in] dgp The DGP graph to update.
 * @param [in] inputs The inputs to the graph.
 * @param [in] reset Whether to reset states to initial values.
 */
void
graph_update(const struct Graph *dgp, const double *inputs, const bool reset)
{
    if (reset) {
        graph_reset(dgp);
    }
    for (int t = 0; t < dgp->t; ++t) {
        synchronous_update(dgp, inputs);
    }
}

/**
 * @brief Prints a DGP graph.
 * @param [in] dgp The DGP graph to print.
 */
void
graph_print(const struct Graph *dgp)
{
    printf("%s\n", graph_json_export(dgp));
}

/**
 * @brief Returns a json formatted string representation of a DGP graph.
 * @param [in] dgp The DGP graph to return.
 * @return String encoded in json format.
 */
const char *
graph_json_export(const struct Graph *dgp)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "n", dgp->n);
    cJSON_AddNumberToObject(json, "t", dgp->t);
    cJSON_AddNumberToObject(json, "n_inputs", dgp->n_inputs);
    cJSON *istate = cJSON_CreateDoubleArray(dgp->initial_state, dgp->n);
    cJSON_AddItemToObject(json, "initial_state", istate);
    cJSON *state = cJSON_CreateDoubleArray(dgp->state, dgp->n);
    cJSON_AddItemToObject(json, "current_state", state);
    cJSON *functions = cJSON_CreateArray();
    cJSON_AddItemToObject(json, "functions", functions);
    for (int i = 0; i < dgp->n; ++i) {
        cJSON *str = cJSON_CreateString(function_string(dgp->function[i]));
        cJSON_AddItemToArray(functions, str);
    }
    cJSON *connectivity = cJSON_CreateIntArray(dgp->connectivity, dgp->klen);
    cJSON_AddItemToObject(json, "connectivity", connectivity);
    cJSON *mutation = cJSON_CreateDoubleArray(dgp->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Frees a DGP graph.
 * @param [in] dgp The DGP graph to be freed.
 */
void
graph_free(const struct Graph *dgp)
{
    free(dgp->connectivity);
    free(dgp->state);
    free(dgp->initial_state);
    free(dgp->tmp_state);
    free(dgp->tmp_input);
    free(dgp->function);
    free(dgp->mu);
}

/**
 * @brief Mutates a specified DGP graph.
 * @param [in] dgp The DGP graph to be mutated.
 * @return Whether any alterations were made.
 */
bool
graph_mutate(struct Graph *dgp)
{
    bool mod = false;
    sam_adapt(dgp->mu, N_MU, MU_TYPE);
    if (graph_mutate_functions(dgp)) {
        mod = true;
    }
    if (graph_mutate_connectivity(dgp)) {
        mod = true;
    }
    if (dgp->evolve_cycles && graph_mutate_cycles(dgp)) {
        mod = true;
    }
    return mod;
}

/**
 * @brief Writes DGP graph to a file.
 * @param [in] dgp The DGP graph to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
graph_save(const struct Graph *dgp, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&dgp->evolve_cycles, sizeof(bool), 1, fp);
    s += fwrite(&dgp->n, sizeof(int), 1, fp);
    s += fwrite(&dgp->t, sizeof(int), 1, fp);
    s += fwrite(&dgp->klen, sizeof(int), 1, fp);
    s += fwrite(&dgp->max_t, sizeof(int), 1, fp);
    s += fwrite(&dgp->max_k, sizeof(int), 1, fp);
    s += fwrite(dgp->state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->function, sizeof(int), dgp->n, fp);
    s += fwrite(dgp->connectivity, sizeof(int), dgp->klen, fp);
    s += fwrite(dgp->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads DGP graph from a file.
 * @param [in] dgp The DGP graph to load.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
graph_load(struct Graph *dgp, FILE *fp)
{
    size_t s = 0;
    s += fread(&dgp->evolve_cycles, sizeof(bool), 1, fp);
    s += fread(&dgp->n, sizeof(int), 1, fp);
    s += fread(&dgp->t, sizeof(int), 1, fp);
    s += fread(&dgp->klen, sizeof(int), 1, fp);
    s += fread(&dgp->max_t, sizeof(int), 1, fp);
    s += fread(&dgp->max_k, sizeof(int), 1, fp);
    if (dgp->n < 1 || dgp->klen < 1) {
        printf("graph_load(): read error\n");
        dgp->n = 1;
        dgp->klen = 1;
        exit(EXIT_FAILURE);
    }
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->tmp_state = malloc(sizeof(double) * dgp->n);
    dgp->tmp_input = malloc(sizeof(double) * dgp->max_k);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
    s += fread(dgp->state, sizeof(double), dgp->n, fp);
    s += fread(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fread(dgp->function, sizeof(int), dgp->n, fp);
    s += fread(dgp->connectivity, sizeof(int), dgp->klen, fp);
    s += fread(dgp->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Sets DGP parameters to default values.
 * @param [in] args Parameters for initialising and operating DGP graphs.
 */
void
graph_args_init(struct ArgsDGP *args)
{
    args->max_k = 0;
    args->max_t = 0;
    args->n = 0;
    args->n_inputs = 0;
    args->evolve_cycles = false;
}

/**
 * @brief Returns a json formatted string of the DGP parameters.
 * @param [in] args Parameters for initialising and operating DGP graphs.
 * @return String encoded in json format.
 */
const char *
graph_args_json_export(const struct ArgsDGP *args)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "max_k", args->max_k);
    cJSON_AddNumberToObject(json, "max_t", args->max_t);
    cJSON_AddNumberToObject(json, "n", args->n);
    cJSON_AddBoolToObject(json, "evolve_cycles", args->evolve_cycles);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Saves DGP parameters.
 * @param [in] args Parameters for initialising and operating DGP graphs.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
graph_args_save(const struct ArgsDGP *args, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&args->evolve_cycles, sizeof(bool), 1, fp);
    s += fwrite(&args->max_k, sizeof(int), 1, fp);
    s += fwrite(&args->max_t, sizeof(int), 1, fp);
    s += fwrite(&args->n, sizeof(int), 1, fp);
    s += fwrite(&args->n_inputs, sizeof(int), 1, fp);
    return s;
}

/**
 * @brief Loads DGP parameters.
 * @param [in] args Parameters for initialising and operating DGP graphs.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
graph_args_load(struct ArgsDGP *args, FILE *fp)
{
    size_t s = 0;
    s += fread(&args->evolve_cycles, sizeof(bool), 1, fp);
    s += fread(&args->max_k, sizeof(int), 1, fp);
    s += fread(&args->max_t, sizeof(int), 1, fp);
    s += fread(&args->n, sizeof(int), 1, fp);
    s += fread(&args->n_inputs, sizeof(int), 1, fp);
    return s;
}

/* parameter setters */

void
graph_param_set_max_k(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP MAX_K too small\n");
        args->max_k = 1;
    } else {
        args->max_k = a;
    }
}

void
graph_param_set_max_t(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP MAX_T too small\n");
        args->max_t = 1;
    } else {
        args->max_t = a;
    }
}

void
graph_param_set_n(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP N too small\n");
        args->n = 1;
    } else {
        args->n = a;
    }
}

void
graph_param_set_n_inputs(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP N_INPUTS too small\n");
        args->n_inputs = 1;
    } else {
        args->n_inputs = a;
    }
}

void
graph_param_set_evolve_cycles(struct ArgsDGP *args, const bool a)
{
    args->evolve_cycles = a;
}
