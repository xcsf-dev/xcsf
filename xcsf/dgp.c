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
 * @date 2016--2019.
 * @brief An implementation of dynamical GP graphs with fuzzy activation functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "dgp.h"

#define NUM_FUNC 6 //!< number of selectable node functions

char *function_string(int function);
double node_activate(int function, double *inputs, int k);

/**
 * @brief Initialises a new DGP graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to initialise.
 * @param n The the number of nodes in the graph.
 */
void graph_init(XCSF *xcsf, GRAPH *dgp, int n)
{
    dgp->t = 0;
    dgp->n = n;
    dgp->klen = n * xcsf->MAX_K;
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
}

/**
 * @brief Copies a DGP graph.
 * @param xcsf The XCSF data structure.
 * @param to The destination DGP graph.
 * @param from The source DGP graph.
 */
void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from)
{ 	
    (void)xcsf;
    to->t = from->t;
    to->n = from->n;
    to->klen = from->klen;
    memcpy(to->state, from->state, sizeof(double) * from->n);
    memcpy(to->initial_state, from->initial_state, sizeof(double) * from->n);
    memcpy(to->function, from->function, sizeof(int) * from->n);
    memcpy(to->connectivity, from->connectivity, sizeof(int) * from->klen);
}

/**
 * @brief Returns the current state of a specified node in the graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to output.
 * @param i Which node within the graph to output.
 * @return The current state of the specified node.
 */
double graph_output(XCSF *xcsf, GRAPH *dgp, int i)
{
    (void)xcsf;
    return dgp->state[i];
}

/**
 * @brief Resets the states to their initial state.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to reset.
 */
void graph_reset(XCSF *xcsf, GRAPH *dgp)
{
    (void)xcsf;
    for(int i = 0; i < dgp->n; i++) {
        dgp->state[i] = dgp->initial_state[i];
    }
}

/**
 * @brief Randomises a specified DGP graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to randomise.
 */
void graph_rand(XCSF *xcsf, GRAPH *dgp)
{
    dgp->t = irand_uniform(1,xcsf->MAX_T);
    for(int i = 0; i < dgp->n; i++) {
        dgp->function[i] = irand_uniform(0, NUM_FUNC);
        dgp->initial_state[i] = rand_normal(0,0.1);
        dgp->state[i] = rand_normal(0,0.1);
    }
    for(int i = 0; i < dgp->klen; i++) {
        // other nodes within the graph
        if(rand_uniform(0,1) < 0.5) {
            dgp->connectivity[i] = irand_uniform(0,dgp->n);
        }
        // external inputs
        else {
            dgp->connectivity[i] = -(irand_uniform(1,xcsf->num_x_vars+1));
        }
    }
}

/**
 * @brief Synchronously updates a DGP graph T cycles.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to update.
 * @param inputs The inputs to the graph.
 */
void graph_update(XCSF *xcsf, GRAPH *dgp, double *inputs)
{
    if(xcsf->RESET_STATES) {
        graph_reset(xcsf, dgp);
    }
    double in[xcsf->MAX_K];
    for(int t = 0; t < dgp->t; t++) {
        // synchronously update each node
        for(int i = dgp->n-1; i >= 0; i--) {
            // each connection
            for(int k = 0; k < xcsf->MAX_K; k++) {
                int idx = (i * xcsf->MAX_K) + k;
                int c = dgp->connectivity[idx];
                // another node within the graph
                if(c >= 0) {
                    in[k] = dgp->state[c];
                }
                // external input
                else {
                    in[k] = inputs[abs(c)-1];
                }
            }
            dgp->state[i] = node_activate(dgp->function[i], in, xcsf->MAX_K);
        }
    }
}

/**
 * @brief Prints a DGP graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to print.
 */
void graph_print(XCSF *xcsf, GRAPH *dgp)
{
    printf("Graph: N=%d; T=%d\n", dgp->n, dgp->t);
    for(int i = 0; i < dgp->n; i++) {
        printf("Node %d: func=%s state=%f init_state=%f con=[", 
                i, function_string(dgp->function[i]),
                dgp->state[i], dgp->initial_state[i]);
        printf("%d", dgp->connectivity[0]);
        for(int j = 1; j < xcsf->MAX_K; j++) {
            printf(",%d", dgp->connectivity[i]);
        }
        printf("]\n");
    }
}

/**
 * @brief Frees a DGP graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to be freed.
 */
void graph_free(XCSF *xcsf, GRAPH *dgp)
{
    (void)xcsf;
    free(dgp->connectivity);
    free(dgp->state);
    free(dgp->initial_state);
    free(dgp->function);
}

/**
 * @brief Mutates a specified DGP graph.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to be mutated.
 * @return True.
 */
_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp)
{
    for(int i = 0; i < dgp->n; i++) {
        // mutate function
        if(rand_uniform(0,1) < xcsf->F_MUTATION) {
            dgp->function[i] = irand_uniform(0, NUM_FUNC);
        }
        // mutate initial state
        dgp->initial_state[i] += rand_normal(0, xcsf->S_MUTATION);
        // mutate connectivity map
        for(int j = 0; j < xcsf->MAX_K; j++) {
            int idx = (i*xcsf->MAX_K)+j;
            if(rand_uniform(0,1) < xcsf->P_MUTATION) {
                // external connection
                if(rand_uniform(0,1) < 0.5) {
                    dgp->connectivity[idx] = -(irand_uniform(1,xcsf->num_x_vars+1));
                }
                // another node
                else {
                    dgp->connectivity[idx] = irand_uniform(0,dgp->n);
                }
            }
        }   
    }               
    // mutate number of update cycles
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        if(rand_uniform(0,1) < 0.5) {
            if(dgp->t > 1) {
                (dgp->t)--;
            }
        }
        else {
            if(dgp->t < xcsf->MAX_T) {
                (dgp->t)++;
            }
        }
    }
    return true;
}

/**
 * @brief Performs uniform crossover with two DGP graphs.
 * @param xcsf The XCSF data structure.
 * @param dgp1 The first DGP graph to perform crossover.
 * @param dgp2 The second DGP graph to perform crossover.
 * @return Whether crossover was performed.
 */
_Bool graph_crossover(XCSF *xcsf, GRAPH *dgp1, GRAPH *dgp2)
{
    if(rand_uniform(0,1) > xcsf->P_CROSSOVER) {
        return false;
    }
    // number of update cycles
    if(rand_uniform(0,1) < 0.5) {
        int tmp = dgp1->t;
        dgp1->t = dgp2->t;
        dgp2->t = tmp;
    }
    for(int i = 0; i < dgp1->n; i++) {
        // functions
        if(rand_uniform(0,1) < 0.5) {
            int tmp = dgp1->function[i];
            dgp1->function[i] = dgp2->function[i];
            dgp2->function[i] = tmp;
        }
        // initial states
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->initial_state[i];
            dgp1->initial_state[i] = dgp2->initial_state[i];
            dgp2->initial_state[i] = tmp;
        }     
        // states
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->state[i];
            dgp1->state[i] = dgp2->state[i];
            dgp2->state[i] = tmp;
        } 
    }
    // connectivity map
    for(int i = 0; i < dgp1->klen; i++) {
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->connectivity[i];
            dgp1->connectivity[i] = dgp2->connectivity[i];
            dgp2->connectivity[i] = tmp;
        }
    }  
    return true;
}

/**
 * @brief Returns the result from applying a specified activation function.
 * @param function The activation function to apply.
 * @param inputs The input to the activation function.
 * @param k The number of inputs to the activation function.
 * @return The result from applying the activation function.
 */
double node_activate(int function, double *inputs, int k)
{
    double state = 0;
    switch(function)
    {
        case 0: // Fuzzy OR (Max/Min)
            state = inputs[0];
            for(int i = 1; i < k; i++) {
                state = fmax(state, inputs[i]);
            }
            break;
        case 1: // Fuzzy AND (CFMQVS and Probabilistic)
            state = inputs[0];
            for(int i = 1; i < k; i++) {
                state *= inputs[i];
            }
            break;
        case 2: // Fuzzy AND (Max/Min)
            state = inputs[0];
            for(int i = 1; i < k; i++) {
                state = fmin(state, inputs[i]);
            }
            break;
        case 3: // Fuzzy OR (CFMQVS and MV)
            state = inputs[0];
            for(int i = 1; i < k; i++) {
                state += inputs[i];
            }
            state = fmin(1, state);
            break;
        case 4: // Fuzzy NOT
            state = 1 - inputs[0];
            break;
        case 5: // Identity
            state = inputs[0];
            break;
        default: // Invalid function
            printf("Error updating node: Invalid function: %d\n", function);
            exit(0);
    }
    state = constrain(0, 1, state);
    return state;
}

/**
 * @brief Returns the name of a specified node function.
 * @param function The node function.
 * @return The name of the node function.
 */
char *function_string(int function)
{
     switch(function) {
        case 0: return "Fuzzy OR (Max/Min)";
        case 1: return "Fuzzy AND (CFMQVS and Probabilistic)";
        case 2: return "Fuzzy AND (Max/Min)";
        case 3: return "Fuzzy OR (CFMQVS and MV)";
        case 4: return "Fuzzy NOT";
        case 5: return "Identity";
        default:
            printf("function_string(): invalid node function: %d\n", function);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Writes DGP graph to a binary file.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to save.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t graph_save(XCSF *xcsf, GRAPH *dgp, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&dgp->n, sizeof(int), 1, fp);
    s += fwrite(&dgp->t, sizeof(int), 1, fp);
    s += fwrite(&dgp->klen, sizeof(int), 1, fp);
    s += fwrite(dgp->state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->function, sizeof(int), dgp->n, fp);
    s += fwrite(dgp->connectivity, sizeof(int), dgp->klen, fp);
    //printf("graph saved %lu elements\n", (unsigned long)s);
    return s;
}

/**
 * @brief Reads DGP graph from a binary file.
 * @param xcsf The XCSF data structure.
 * @param dgp The DGP graph to load.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t graph_load(XCSF *xcsf, GRAPH *dgp, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&dgp->n, sizeof(int), 1, fp);
    s += fread(&dgp->t, sizeof(int), 1, fp);
    s += fread(&dgp->klen, sizeof(int), 1, fp);
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
    s += fread(dgp->state, sizeof(double), dgp->n, fp);
    s += fread(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fread(dgp->function, sizeof(int), dgp->n, fp);
    s += fread(dgp->connectivity, sizeof(int), dgp->klen, fp);
    //printf("graph loaded %lu elements\n", (unsigned long)s);
    return s;
}
