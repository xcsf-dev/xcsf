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
 * The constants module.
 *
 * Reads in the global constants from cons.txt. The number of trials and number
 * of experiments to perform can be overridden by passing command line
 * arguments.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "cons.h"
#include "gp.h"

#define MAXLEN 127
typedef char *pchar;
typedef struct nv *pnv;
typedef struct section *psection;

struct nv {
    pchar name;
    pchar value;
    pnv next;
};

struct section {
    pchar name;
    pnv nvlist;
    psection next;
};

void init_config(pchar filename);
int isname(pchar section,pchar name);
void tidyup();
pchar getvalue(pchar name);     

psection head;
psection current;

void constants_init()
{
    init_config("cons.txt");
    COND_TYPE = atoi(getvalue("COND_TYPE"));
    PRED_TYPE = atoi(getvalue("PRED_TYPE"));
    POP_SIZE = atoi(getvalue("POP_SIZE"));
    if(strcmp(getvalue("POP_INIT"), "false") == 0) {
        POP_INIT = false;
    }
    else {
        POP_INIT = true;
    }
    NUM_EXPERIMENTS = atoi(getvalue("NUM_EXPERIMENTS"));
    MAX_TRIALS = atoi(getvalue("MAX_TRIALS"));
    P_CROSSOVER = atof(getvalue("P_CROSSOVER"));
    P_MUTATION = atof(getvalue("P_MUTATION"));
    THETA_SUB = atof(getvalue("THETA_SUB"));
    EPS_0 = atof(getvalue("EPS_0"));
    DELTA = atof(getvalue("DELTA"));
    THETA_DEL = atof(getvalue("THETA_DEL"));
    THETA_GA = atof(getvalue("THETA_GA"));
    THETA_MNA = atoi(getvalue("THETA_MNA"));
    THETA_OFFSPRING = atoi(getvalue("THETA_OFFSPRING"));
    BETA = atof(getvalue("BETA"));
    ALPHA = atof(getvalue("ALPHA")); 
    NU = atof(getvalue("NU"));
    INIT_FITNESS = atof(getvalue("INIT_FITNESS"));
    INIT_ERROR = atof(getvalue("INIT_ERROR"));
    ERR_REDUC = atof(getvalue("ERR_REDUC"));
    FIT_REDUC = atof(getvalue("FIT_REDUC"));
    if(strcmp(getvalue("GA_SUBSUMPTION"), "false") == 0) {
        GA_SUBSUMPTION = false;
    }
    else {
        GA_SUBSUMPTION = true;
    }
    if(strcmp(getvalue("SET_SUBSUMPTION"), "false") == 0) {
        SET_SUBSUMPTION = false;
    }
    else {
        SET_SUBSUMPTION = true;
    }
    PERF_AVG_TRIALS = atoi(getvalue("PERF_AVG_TRIALS"));
    XCSF_X0 = atof(getvalue("XCSF_X0"));
    XCSF_ETA = atof(getvalue("XCSF_ETA"));
    muEPS_0 = atof(getvalue("muEPS_0"));
    NUM_SAM = atoi(getvalue("NUM_SAM"));
    S_MUTATION = atof(getvalue("S_MUTATION"));
    MIN_CON = atof(getvalue("MIN_CON"));
    MAX_CON = atof(getvalue("MAX_CON"));
    NUM_HIDDEN_NEURONS = atoi(getvalue("NUM_HIDDEN_NEURONS"));
    DGP_NUM_NODES = atoi(getvalue("DGP_NUM_NODES"));
    tidyup();  

    tree_init_cons();
} 
void trim(pchar s) // Remove tabs/spaces/lf/cr  both ends
{
    size_t i=0,j;
    while((s[i]==' ' || s[i]=='\t' || s[i] =='\n' || s[i]=='\r'))
        i++;
    if(i>0) {
        for( j=0; j < strlen(s);j++)
            s[j]=s[j+i];
        s[j]='\0';
    }
    i=strlen(s)-1;
    while((s[i]==' ' || s[i]=='\t'|| s[i] =='\n' || s[i]=='\r'))
        i--;
    if(i < (strlen(s)-1))
        s[i+1]='\0';
}

void newsection(pchar config) {
    psection newsect = malloc(sizeof(struct section));
    if(head == NULL)
        head = newsect;
    else
        current->next = newsect;
    current = newsect;
    newsect->name = malloc(strlen(config));
    strncpy(newsect->name,config+1,strlen(config)-1);
    newsect->name[strlen(config)-2]= '\0';
    newsect->nvlist = NULL; 
    newsect->next   = NULL;
}

void newnvpair(pchar config) {
    pchar name= NULL;
    pchar value = NULL;
    pnv newnv = NULL;
    pnv lastnv;
    size_t valuelen;
    size_t p=0;
    int err=2;
    if(current==NULL)
        exit(1);
    for(p=0; (p < strlen(config)) ;p++) {
        if (config[p]=='=' ) {
            err=0;
            break;
        }
    }
    if(err==2)
        exit(2);
    newnv = malloc(sizeof(struct nv));
    name=malloc(p+1);
    strncpy(name,config,p);
    name[p]='\0';
    valuelen = strlen(config)-p-1;
    value= malloc(valuelen+1);
    strncpy(value,config+p+1,valuelen );
    value[valuelen]='\0';
    newnv->name = name;
    newnv->value = value;
    newnv->next = NULL;
    if(current->nvlist == NULL)
        current->nvlist = newnv;
    else {
        lastnv= current->nvlist;
        while((lastnv->next ) != NULL)
            lastnv=lastnv->next;
        lastnv->next = newnv;
    }
}

psection findsection(pchar sectionname) {
    psection result = NULL;
    current = head;
    while(current) {
        if(strcmp(current->name,sectionname)==0) {
            result=current;
            break;
        }
        current = current->next;
    }
    return result;
}

pchar getvalue(pchar name) {
    pchar result = NULL;
    pnv currnv = current->nvlist;
    while(currnv) {
        if((strcmp(name,currnv->name)== 0 )) {
            result = currnv->value;
            break;
        }
        currnv = currnv->next;
    }
    return result;
}

void process(pchar configline) {
    if(strlen(configline)== 0) // ignore empty lines
        return;
    if(configline[0]==';')  // lines starting with a ; are comments
        return; 
    if(configline[0]=='[') 
        newsection(configline);
    else 
        newnvpair(configline);
}

int isname(pchar section,pchar name) {
    int result = 0;
    trim(section);
    current = findsection(section);
    if(current) {
        if(getvalue(name))
            result =1;
    }
    return result;
}

void init_config(pchar filename) {
    FILE * f;
    char buff[MAXLEN];
    f = fopen(filename,"rt");
    if(f==NULL)
        return;
    head = NULL;
    while(!feof(f)) {
        if(fgets(buff,MAXLEN-2,f)==NULL)
            break;
        trim(buff);
        process(buff);
    }
    fclose(f);
}

void tidyup()
{	
    pnv currentnv;
    pnv	nextnv;
    psection nextsection;
    current = head;
    do  {
        currentnv = current->nvlist;
        do {
            if(currentnv) {					
                nextnv = currentnv->next;				
                free(currentnv->value);
                free(currentnv->name);
                free(currentnv);
                currentnv = nextnv;
            } 
        }
        while(currentnv);
        if(current) {	
            nextsection = current->next;
            free(current->name);
            free(current);
            current = nextsection;
        }
    }
    while(current);
    head=NULL;
}
