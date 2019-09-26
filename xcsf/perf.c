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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <errno.h>
#include "xcsf.h"
#include "cl.h"
#include "cl_set.h"
#include "perf.h"

void disp_perf1(XCSF *xcsf, double error, int trial)
{
    printf("%d %.5f %d", trial, error, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        printf(" %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
    }
    printf("\n");    
    fflush(stdout);
#ifdef GNUPLOT
    gplot_perf1(xcsf, error, trial);
#endif
}          

void disp_perf2(XCSF *xcsf, double error, double terror, int trial)
{
    printf("%d %.5f %.5f %d", trial, error, terror, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        printf(" %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
    }
    printf("\n");    
    fflush(stdout);
#ifdef GNUPLOT
    gplot_perf2(xcsf, error, trial);
#endif
}          
 
#ifndef GNUPLOT

void gplot_init(XCSF *xcsf)
{
    (void)xcsf;
}

void gplot_free(XCSF *xcsf)
{
    (void)xcsf;
}

#else

FILE *gp; // file containing gnuplot script
FILE *fout; // file containing performance data
char fname[50]; // file name for performance data
void gplot_draw(XCSF *xcsf, _Bool test_error);
 
void gplot_perf1(XCSF *xcsf, double error, int trial)
{
    fprintf(fout, "%d %.5f %d", trial, error, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        fprintf(fout, " %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
    }
    fprintf(fout, "\n");
    fflush(fout);
    gplot_draw(xcsf, false); 
}
 
void gplot_perf2(XCSF *xcsf, double error, double terror, int trial)
{
    fprintf(fout, "%d %.5f %.5f %d", trial, error, terror, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        fprintf(fout, " %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
    }
    fprintf(fout, "\n");
    fflush(fout);
    gplot_draw(xcsf, true); 
}

void gplot_init(XCSF *xcsf)
{ 	
    // file name for writing performance uses the current date-time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(fname, "out/%04d-%02d-%02d-%02d%02d%02d.dat", tm.tm_year + 1900, 
            tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec); 

    // create file for writing performance
    fout = fopen(fname, "wt");
    if(fout == 0) {
        printf("Error opening file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }       

    // set gnuplot title
    char buffer[20];
    char title[200];
    title[0] = '\0';

    switch(xcsf->COND_TYPE) {
        case -1:
            strcat(title, " dummy cond");
            break;
        case 0:
            strcat(title, " hyperrectangle cond");
            break;
        case 1:
            strcat(title, " hyperellipsoid cond");
            break;
        case 2:
            strcat(title, " neural cond");
            break;
        case 3:
            strcat(title, " tree-GP cond");
            break;
        case 4:
            strcat(title, " graph-DGP cond");
            break;
        case 11:
            strcat(title, " graph-DGP rules");
            break;
        case 12:
            strcat(title, " neural rules");
            break;
    }

    if(xcsf->COND_TYPE < 10) {
        switch(xcsf->PRED_TYPE) {
            case 0:
                strcat(title, ", linear nlms");
                break;
            case 1:
                strcat(title, ", quadratic nlms");
                break;
            case 2:
                strcat(title, ", linear rls");
                break;
            case 3:
                strcat(title, ", quadratic rls");
                break;
            case 4:
                strcat(title, ", neural pred");
                break;
        }
    }

    if(xcsf->SAM_NUM > 0) {
        strcat(title, ", SAM");
    }

    sprintf(buffer, ", P=%d", xcsf->POP_SIZE);
    strcat(title, buffer);

    // execute gnuplot
#ifdef _WIN32
    gp = _popen("C:\Program Files (x86)\gnuplot\bin\pgnuplot.exe -persistent", "w");
#else
    gp = popen("gnuplot -persistent", "w");
#endif
    if(gp != NULL) {
        fprintf(gp, "set terminal wxt noraise enhanced font 'Arial,12'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "set border linewidth 1\n");
        fprintf(gp, "set title \"%s\"\n", title);
        //fprintf(gp, "set nokey\n");
        fprintf(gp, "set xlabel 'trials'\n");
        fprintf(gp, "set ylabel 'system error'\n");
        fprintf(gp, "set style line 1 lt -1 lw 1 ps 1 lc rgb 'red'\n");
        fprintf(gp, "set style line 2 lt -1 lw 1 ps 1 lc rgb 'blue'\n");
    }
    else {
        printf("error starting gnuplot\n");
    }
}

void gplot_free(XCSF *xcsf)
{
    (void)xcsf;
    // close gnuplot
    if(gp != NULL) {
        pclose(gp);
    }
    else {
        printf("error closing gnuplot\n");
    }
    // close data file
    fclose(fout);
}

void gplot_draw(XCSF *xcsf, _Bool test_error)
{
    if(gp != NULL) {
        fprintf(gp, "plot '%s' using 1:2 title 'train error' w lp ls 1 pt 4 pi 50, ", fname);
        if(test_error) {
            fprintf(gp, "'%s' using 1:3 title 'test error' w lp ls 2 pt 8 pi 50", fname);
        }
        fprintf(gp,"\nreplot\n");
        fflush(gp);
    }
    else {
        printf("error writing to gnuplot\n");
    }
    (void)xcsf;
}
#endif
