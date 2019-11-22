/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 */
                
/**
 * @file config.c
 * @brief Config file handling functions
 */ 
 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include "xcsf.h"
#include "gp.h"
#include "config.h"
#include "loss.h"

#define MAXLEN 127

/**
 * @brief Config file parameter data structure.
 */ 
typedef struct nv {
    char *name; //!< parameter name
    char *value; //!< parameter value
    struct nv *next; //!< pointer to the next parameter
} nv;

nv *head;

void init_config(const char *filename);
void process(char *configline);
void trim(char *s);
void newnvpair(const char *config);
char *getvalue(char *name);
void tidyup();

void constants_init(XCSF *xcsf, const char *filename)
{
    init_config(filename);
    // integers
    xcsf->ACT_TYPE = atoi(getvalue("ACT_TYPE"));
    xcsf->COND_TYPE = atoi(getvalue("COND_TYPE"));
    xcsf->DGP_NUM_NODES = atoi(getvalue("DGP_NUM_NODES"));
    xcsf->GP_INIT_DEPTH = atoi(getvalue("GP_INIT_DEPTH"));
    xcsf->GP_NUM_CONS = atoi(getvalue("GP_NUM_CONS"));
    xcsf->LOSS_FUNC = atoi(getvalue("LOSS_FUNC"));
    xcsf->MAX_K = atoi(getvalue("MAX_K"));
    xcsf->MAX_T = atoi(getvalue("MAX_T"));
    xcsf->MAX_TRIALS = atoi(getvalue("MAX_TRIALS"));
    xcsf->COND_NUM_HIDDEN_NEURONS = atoi(getvalue("COND_NUM_HIDDEN_NEURONS"));
    xcsf->COND_MAX_HIDDEN_NEURONS = atoi(getvalue("COND_MAX_HIDDEN_NEURONS"));
    xcsf->COND_HIDDEN_NEURON_ACTIVATION = atoi(getvalue("COND_HIDDEN_NEURON_ACTIVATION"));
    xcsf->PRED_NUM_HIDDEN_NEURONS = atoi(getvalue("PRED_NUM_HIDDEN_NEURONS"));
    xcsf->PRED_MAX_HIDDEN_NEURONS = atoi(getvalue("PRED_MAX_HIDDEN_NEURONS"));
    xcsf->PRED_HIDDEN_NEURON_ACTIVATION = atoi(getvalue("PRED_HIDDEN_NEURON_ACTIVATION"));
    xcsf->OMP_NUM_THREADS = atoi(getvalue("OMP_NUM_THREADS"));
    xcsf->PERF_AVG_TRIALS = atoi(getvalue("PERF_AVG_TRIALS"));
    xcsf->POP_SIZE = atoi(getvalue("POP_SIZE"));
    xcsf->PRED_TYPE = atoi(getvalue("PRED_TYPE"));
    xcsf->SAM_NUM = atoi(getvalue("SAM_NUM"));
    xcsf->SAM_TYPE = atoi(getvalue("SAM_TYPE"));
    xcsf->THETA_MNA = atoi(getvalue("THETA_MNA"));
    xcsf->LAMBDA = atoi(getvalue("LAMBDA"));
    xcsf->EA_SELECT_TYPE = atoi(getvalue("EA_SELECT_TYPE"));
    xcsf->THETA_SUB = atoi(getvalue("THETA_SUB"));
    xcsf->THETA_DEL = atoi(getvalue("THETA_DEL"));
    // floats
    xcsf->ALPHA = atof(getvalue("ALPHA")); 
    xcsf->BETA = atof(getvalue("BETA"));
    xcsf->DELTA = atof(getvalue("DELTA"));
    xcsf->EPS_0 = atof(getvalue("EPS_0"));
    xcsf->ERR_REDUC = atof(getvalue("ERR_REDUC"));
    xcsf->FIT_REDUC = atof(getvalue("FIT_REDUC"));
    xcsf->INIT_ERROR = atof(getvalue("INIT_ERROR"));
    xcsf->INIT_FITNESS = atof(getvalue("INIT_FITNESS"));
    xcsf->NU = atof(getvalue("NU"));
    xcsf->THETA_EA = atof(getvalue("THETA_EA"));
    xcsf->EA_SELECT_SIZE = atof(getvalue("EA_SELECT_SIZE"));
    xcsf->P_CROSSOVER = atof(getvalue("P_CROSSOVER"));
    xcsf->F_MUTATION = atof(getvalue("F_MUTATION"));
    xcsf->P_MUTATION = atof(getvalue("P_MUTATION"));
    xcsf->S_MUTATION = atof(getvalue("S_MUTATION"));
    xcsf->E_MUTATION = atof(getvalue("E_MUTATION"));
    xcsf->SAM_MIN = atof(getvalue("SAM_MIN"));
    xcsf->MAX_CON = atof(getvalue("MAX_CON"));
    xcsf->MIN_CON = atof(getvalue("MIN_CON"));
    xcsf->COND_ETA = atof(getvalue("COND_ETA"));
    xcsf->PRED_RLS_LAMBDA = atof(getvalue("PRED_RLS_LAMBDA"));
    xcsf->PRED_RLS_SCALE_FACTOR = atof(getvalue("PRED_RLS_SCALE_FACTOR"));
    xcsf->PRED_X0 = atof(getvalue("PRED_X0"));
    xcsf->PRED_ETA = atof(getvalue("PRED_ETA"));
    xcsf->PRED_MOMENTUM = atof(getvalue("PRED_MOMENTUM"));
    // Bools
    xcsf->POP_INIT = false;
    if(strncmp(getvalue("POP_INIT"), "true", 4) == 0) {
        xcsf->POP_INIT = true;
    }
    xcsf->EA_SUBSUMPTION = false;
    if(strncmp(getvalue("EA_SUBSUMPTION"), "true", 4) == 0) {
        xcsf->EA_SUBSUMPTION = true;
    }
    xcsf->SET_SUBSUMPTION = false;
    if(strncmp(getvalue("SET_SUBSUMPTION"), "true", 4) == 0) {
        xcsf->SET_SUBSUMPTION = true;
    }
    xcsf->RESET_STATES = false;
    if(strncmp(getvalue("RESET_STATES"), "true", 4) == 0) {
        xcsf->RESET_STATES = true;
    }
    xcsf->COND_EVOLVE_WEIGHTS = false;
    if(strncmp(getvalue("COND_EVOLVE_WEIGHTS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_WEIGHTS = true;
    }
    xcsf->COND_EVOLVE_NEURONS = false;
    if(strncmp(getvalue("COND_EVOLVE_NEURONS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_NEURONS = true;
    }
    xcsf->COND_EVOLVE_FUNCTIONS = false;
    if(strncmp(getvalue("COND_EVOLVE_FUNCTIONS"), "true", 4) == 0) {
        xcsf->COND_EVOLVE_FUNCTIONS = true;
    }
    xcsf->PRED_EVOLVE_WEIGHTS = false;
    if(strncmp(getvalue("PRED_EVOLVE_WEIGHTS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_WEIGHTS = true;
    }
    xcsf->PRED_EVOLVE_NEURONS = false;
    if(strncmp(getvalue("PRED_EVOLVE_NEURONS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_NEURONS = true;
    }
    xcsf->PRED_EVOLVE_FUNCTIONS = false;
    if(strncmp(getvalue("PRED_EVOLVE_FUNCTIONS"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_FUNCTIONS = true;
    }
    xcsf->PRED_EVOLVE_ETA = false;
    if(strncmp(getvalue("PRED_EVOLVE_ETA"), "true", 4) == 0) {
        xcsf->PRED_EVOLVE_ETA = true;
    }
    xcsf->PRED_SGD_WEIGHTS = false;
    if(strncmp(getvalue("PRED_SGD_WEIGHTS"), "true", 4) == 0) {
        xcsf->PRED_SGD_WEIGHTS = true;
    }
    // initialise (shared) tree-GP constants
    tree_init_cons(xcsf);
    // initialise loss/error function
    loss_set_func(xcsf);
    // clean up
    tidyup();
}

void constants_free(XCSF *xcsf) 
{
    tree_free_cons(xcsf);
}

void trim(char *s)
{
    // remove tabs/spaces/lf/cr both ends
    size_t i = 0;
    while((s[i]==' ' || s[i]=='\t' || s[i] =='\n' || s[i]=='\r')) {
        i++;
    }
    if(i > 0) {
        size_t j = 0;
        while(j < strnlen(s, MAXLEN)) {
            s[j] = s[j+i];
            j++;
        }
        s[j] = '\0';
    }
    i = strnlen(s, MAXLEN)-1;
    while((s[i]==' ' || s[i]=='\t'|| s[i] =='\n' || s[i]=='\r')) {
        i--;
    }
    if(i < (strnlen(s, MAXLEN)-1)) {
        s[i+1] = '\0';
    }
}

void newnvpair(const char *config) {
    // first pair
    if(head == NULL) {
        head = malloc(sizeof(nv));
        head->next = NULL;
    }
    // other pairs
    else {
        nv *new = malloc(sizeof(nv));
        new->next = head;
        head = new;
    }
    // get length of name
    size_t namelen = 0; // length of name
    _Bool err = true;
    for(namelen = 0; namelen < strnlen(config, MAXLEN); namelen++) {
        if(config[namelen] == '=') {
            err = false;
            break;
        }
    }
    // no = found
    if(err) {
        printf("error reading config: no '=' found\n");
        exit(EXIT_FAILURE);
    }
    // get name
    char *name = malloc(namelen+1);
    snprintf(name, namelen+1, "%s", config);
    // get value
    size_t valuelen = strnlen(config,MAXLEN)-namelen; // length of value
    char *value = malloc(valuelen);
    snprintf(value, valuelen, "%s", config+namelen+1);
    // add pair
    head->name = name;
    head->value = value;
}

char *getvalue(char *name) {
    char *result = NULL;
    for(nv *iter = head; iter != NULL; iter = iter->next) {
        if(strcmp(name, iter->name) == 0) {
            result = iter->value;
            break;
        }
    }
    return result;
}

void process(char *configline) {
    // ignore empty lines
    if(strnlen(configline, MAXLEN) == 0) {
        return;
    }
    // lines starting with # are comments
    if(configline[0] == '#') {
        return; 
    }
    // remove anything after #
    char *ptr = strchr(configline, '#');
    if(ptr != NULL) {
        *ptr = '\0';
    }
    newnvpair(configline);
}

void init_config(const char *filename) {
    FILE *f = fopen(filename, "rt");
    if(f == NULL) {
        printf("ERROR: cannot open %s\n", filename);
        return;
    }
    char buff[MAXLEN];
    head = NULL;
    while(!feof(f)) {
        if(fgets(buff, MAXLEN-2, f) == NULL) {
            break;
        }
        trim(buff);
        process(buff);
    }
    fclose(f);
}

void tidyup()
{ 
    nv *iter = head;
    while(iter != NULL) {
        free(head->value);
        free(head->name);
        head = iter->next;
        free(iter);
        iter = head;
    }    
    head = NULL;
}
