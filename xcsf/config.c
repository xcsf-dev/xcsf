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
 **************
 * Description: 
 **************
 * Reads the XCSF parameters from a configuration file.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "data_structures.h"
#include "config.h"
#include "gp.h"

#define MAXLEN 127
typedef struct nv {
    char *name;
    char *value;
    struct nv *next;
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
    xcsf->OMP_NUM_THREADS = atoi(getvalue("OMP_NUM_THREADS"));
    xcsf->COND_TYPE = atoi(getvalue("COND_TYPE"));
    xcsf->PRED_TYPE = atoi(getvalue("PRED_TYPE"));
    xcsf->POP_SIZE = atoi(getvalue("POP_SIZE"));
    if(strcmp(getvalue("POP_INIT"), "false") == 0) {
        xcsf->POP_INIT = false;
    }
    else {
        xcsf->POP_INIT = true;
    }
    xcsf->MAX_TRIALS = atoi(getvalue("MAX_TRIALS"));
    xcsf->P_CROSSOVER = atof(getvalue("P_CROSSOVER"));
    xcsf->P_MUTATION = atof(getvalue("P_MUTATION"));
    xcsf->THETA_SUB = atof(getvalue("THETA_SUB"));
    xcsf->EPS_0 = atof(getvalue("EPS_0"));
    xcsf->DELTA = atof(getvalue("DELTA"));
    xcsf->THETA_DEL = atof(getvalue("THETA_DEL"));
    xcsf->THETA_GA = atof(getvalue("THETA_GA"));
    xcsf->THETA_MNA = atoi(getvalue("THETA_MNA"));
    xcsf->THETA_OFFSPRING = atoi(getvalue("THETA_OFFSPRING"));
    xcsf->BETA = atof(getvalue("BETA"));
    xcsf->ALPHA = atof(getvalue("ALPHA")); 
    xcsf->NU = atof(getvalue("NU"));
    xcsf->INIT_FITNESS = atof(getvalue("INIT_FITNESS"));
    xcsf->INIT_ERROR = atof(getvalue("INIT_ERROR"));
    xcsf->ERR_REDUC = atof(getvalue("ERR_REDUC"));
    xcsf->FIT_REDUC = atof(getvalue("FIT_REDUC"));
    if(strcmp(getvalue("GA_SUBSUMPTION"), "false") == 0) {
        xcsf->GA_SUBSUMPTION = false;
    }
    else {
        xcsf->GA_SUBSUMPTION = true;
    }
    if(strcmp(getvalue("SET_SUBSUMPTION"), "false") == 0) {
        xcsf->SET_SUBSUMPTION = false;
    }
    else {
        xcsf->SET_SUBSUMPTION = true;
    }
    xcsf->PERF_AVG_TRIALS = atoi(getvalue("PERF_AVG_TRIALS"));
    xcsf->XCSF_X0 = atof(getvalue("XCSF_X0"));
    xcsf->XCSF_ETA = atof(getvalue("XCSF_ETA"));
    xcsf->RLS_SCALE_FACTOR = atof(getvalue("RLS_SCALE_FACTOR"));
    xcsf->RLS_LAMBDA = atof(getvalue("RLS_LAMBDA"));
    xcsf->muEPS_0 = atof(getvalue("muEPS_0"));
    xcsf->NUM_SAM = atoi(getvalue("NUM_SAM"));
    xcsf->S_MUTATION = atof(getvalue("S_MUTATION"));
    xcsf->MIN_CON = atof(getvalue("MIN_CON"));
    xcsf->MAX_CON = atof(getvalue("MAX_CON"));
    xcsf->NUM_HIDDEN_NEURONS = atoi(getvalue("NUM_HIDDEN_NEURONS"));
    xcsf->MOMENTUM = atof(getvalue("MOMENTUM"));
    xcsf->HIDDEN_NEURON_ACTIVATION = atoi(getvalue("HIDDEN_NEURON_ACTIVATION"));
    xcsf->DGP_NUM_NODES = atoi(getvalue("DGP_NUM_NODES"));
    if(strcmp(getvalue("RESET_STATES"), "false") == 0) {
        xcsf->RESET_STATES = false;
    }
    else {
        xcsf->RESET_STATES = true;
    }
    xcsf->MAX_K = atoi(getvalue("MAX_K"));
    xcsf->MAX_T = atoi(getvalue("MAX_T"));
    xcsf->GP_NUM_CONS = atoi(getvalue("GP_NUM_CONS"));
    xcsf->GP_INIT_DEPTH = atoi(getvalue("GP_INIT_DEPTH"));
    tidyup();  

    tree_init_cons(xcsf);
} 

void constants_free(XCSF *xcsf) 
{
    tree_free_cons(xcsf);
}

void trim(char *s) // Remove tabs/spaces/lf/cr both ends
{
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
    int err = 2;
    for(namelen = 0; namelen < strnlen(config, MAXLEN); namelen++) {
        if(config[namelen] == '=') {
            err = 0;
            break;
        }
    }
    // no = found
    if(err == 2) {
        exit(2);
    }
    // get name
    char *name = malloc(namelen+1);
    snprintf(name, namelen+1, "%s", config);
    // get value
    size_t valuelen = strnlen(config,MAXLEN)-namelen-1; // length of value
    char *value = malloc(valuelen+1);
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
    if(strnlen(configline,MAXLEN) == 0) { // ignore empty lines
        return;
    }
    if(configline[0] == '#') {  // lines starting with # are comments
        return; 
    }
    // remove anything after #
    char *ptr = strchr(configline, '#');
    if (ptr != NULL) {
        *ptr = '\0';
    }
    newnvpair(configline);
}

void init_config(const char *filename) {
    FILE * f;
    char buff[MAXLEN];
    f = fopen(filename,"rt");
    if(f == NULL) {
        printf("ERROR: cannot open %s\n", filename);
        return;
    }
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
