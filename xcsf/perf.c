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
 * @file perf.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief System performance printing and plotting with Gnuplot.
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

#ifndef GNUPLOT

/**
 * @brief Dummy init function if Gnuplot not defined.
 * @param xcsf The XCSF data structure.
 */
void gplot_init(XCSF *xcsf)
{
    (void)xcsf;
}

/**
 * @brief Dummy free function if Gnuplot not defined.
 * @param xcsf The XCSF data structure.
 */
void gplot_free(XCSF *xcsf)
{
    (void)xcsf;
}

#else

static FILE *gp; //!< File containing gnuplot script
static FILE *fout; //!< File containing performance data
static char fname[50]; //!< File name for performance data
static void gplot_perf1(XCSF *xcsf, double error, int trial);
static void gplot_perf2(XCSF *xcsf, double error, double terror, int trial);
static void gplot_draw(XCSF *xcsf, _Bool test_error);
static void gplot_title(XCSF *xcsf, char *title);

/**
 * @brief Writes training performance to a file and redraws Gnuplot.
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param trial The number of learning trials executed.
 */
static void gplot_perf1(XCSF *xcsf, double error, int trial)
{
    fprintf(fout, "%d %.5f %d", trial, error, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        fprintf(fout, " %.5f", set_mean_mut(xcsf, &xcsf->pset, i));
    }
    fprintf(fout, "\n");
    fflush(fout);
    gplot_draw(xcsf, false); 
}

/**
 * @brief Writes training and test performance to a file and redraws Gnuplot.
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param error The current testing error.
 * @param trial The number of learning trials executed.
 */
static void gplot_perf2(XCSF *xcsf, double error, double terror, int trial)
{
    fprintf(fout, "%d %.5f %.5f %d", trial, error, terror, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        fprintf(fout, " %.5f", set_mean_mut(xcsf, &xcsf->pset, i));
    }
    fprintf(fout, "\n");
    fflush(fout);
    gplot_draw(xcsf, true); 
}

/**
 * @brief Initialises Gnuplot and file for writing performance.
 * @param xcsf The XCSF data structure.
 */
void gplot_init(XCSF *xcsf)
{ 	
    // generate file name for writing performance based on the current date-time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(fname, "out/%04d-%02d-%02d-%02d%02d%02d.dat", tm.tm_year + 1900, 
            tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec); 
    // generate Gnuplot title
    char title[100];
    gplot_title(xcsf, title);
    // open file for writing performance
    fout = fopen(fname, "wt");
    if(fout == 0) {
        printf("Error opening file: %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }       
    // execute Gnuplot
    gp = popen("gnuplot -persistent", "w");
    if(gp != NULL) {
        fprintf(gp, "set terminal wxt noraise enhanced font 'Arial,12'\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "set border linewidth 1\n");
        fprintf(gp, "set title \"%s\"\n", title);
        fprintf(gp, "set xlabel 'Trials'\n");
        if(xcsf->num_actions < 2) {
            fprintf(gp, "set ylabel 'System Error'\n");
        }
        fprintf(gp, "set style line 1 lt -1 lw 1 ps 1 lc rgb 'red'\n");
        fprintf(gp, "set style line 2 lt -1 lw 1 ps 1 lc rgb 'blue'\n");
    }
    else {
        printf("Error starting gnuplot\n");
    }
}

/**
 * @brief Generates the title for Gnuplot based on the knowledge representation.
 * @param xcsf The XCSF data structure.
 * @param title The Gnuplot title (set by this function).
 */
static void gplot_title(XCSF *xcsf, char *title)
{
    char buffer[20];
    title[0] = '\0';
    switch(xcsf->COND_TYPE) {
        case COND_TYPE_DUMMY: strcat(title, " dummy cond"); break;
        case COND_TYPE_HYPERRECTANGLE: strcat(title, " hyperrectangle cond"); break;
        case COND_TYPE_HYPERELLIPSOID: strcat(title, " hyperellipsoid cond"); break;
        case COND_TYPE_NEURAL: strcat(title, " neural cond"); break;
        case COND_TYPE_GP: strcat(title, " tree-GP cond"); break;
        case COND_TYPE_DGP: strcat(title, " graph-DGP cond"); break;
        case COND_TYPE_TERNARY: strcat(title, " ternary cond"); break;
        case RULE_TYPE_DGP: strcat(title, " graph-DGP rules"); break;
        case RULE_TYPE_NEURAL: strcat(title, " neural rules"); break;
    }
    if(xcsf->COND_TYPE < RULE_TYPE_DGP) {
        strcat(title, ", action integer");
    }
    switch(xcsf->PRED_TYPE) {
        case PRED_TYPE_CONSTANT: strcat(title, ", constant pred"); break;
        case PRED_TYPE_NLMS_LINEAR: strcat(title, ", linear nlms"); break;
        case PRED_TYPE_NLMS_QUADRATIC: strcat(title, ", quadratic nlms"); break;
        case PRED_TYPE_RLS_LINEAR: strcat(title, ", linear rls"); break;
        case PRED_TYPE_RLS_QUADRATIC: strcat(title, ", quadratic rls"); break;
        case PRED_TYPE_NEURAL: strcat(title, ", neural pred"); break;
    }
    if(xcsf->SAM_NUM > 0) {
        strcat(title, ", SAM");
    }
    sprintf(buffer, ", P=%d", xcsf->POP_SIZE);
    strcat(title, buffer);
}

/**
 * @brief Closes any files and frees any memory used for Gnuplot.
 * @param xcsf The XCSF data structure.
 */
void gplot_free(XCSF *xcsf)
{
    (void)xcsf;
    if(gp != NULL) {
        pclose(gp);
    }
    else {
        printf("Error closing gnuplot\n");
    }
    fclose(fout);
}

/**
 * @brief Draws current performance with Gnuplot.
 * @param xcsf The XCSF data structure.
 * @param test_error Whether to plot a second line.
 */
static void gplot_draw(XCSF *xcsf, _Bool test_error)
{
    (void)xcsf;
    if(gp != NULL) {
        if(xcsf->num_actions < 2) {
            // regression
            fprintf(gp, "plot '%s' using 1:2 title 'Train Error' w lp ls 1 pt 4 pi 50, ", fname);
            if(test_error) {
                fprintf(gp, "'%s' using 1:3 title 'Test Error' w lp ls 2 pt 8 pi 50", fname);
            }
        }
        else {
            // reinforcement learning
            fprintf(gp, "plot '%s' using 1:2 title 'Performance' w lp ls 1 pt 4 pi 50, ", fname);
            if(test_error) {
                fprintf(gp, "'%s' using 1:3 title 'Error' w lp ls 2 pt 8 pi 50", fname);
            }
        }
        fprintf(gp,"\nreplot\n");
        fflush(gp);
    }
    else {
        printf("Error writing to gnuplot\n");
    }
}
#endif

/**
 * @brief Displays the current training performance
 * (additionally redraws Gnuplot if defined.)
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param trial The number of learning trials executed.
 */
void disp_perf1(XCSF *xcsf, double error, int trial)
{
    printf("%d %.5f %d", trial, error, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        printf(" %.5f", set_mean_mut(xcsf, &xcsf->pset, i));
    }
    printf("\n");
    fflush(stdout);
#ifdef GNUPLOT
    gplot_perf1(xcsf, error, trial);
#endif
}

/**
 * @brief Displays the current training and test performance
 * (additionally redraws Gnuplot if defined.)
 * @param xcsf The XCSF data structure.
 * @param error The current training error.
 * @param terror The current testing error.
 * @param trial The number of learning trials executed.
 */
void disp_perf2(XCSF *xcsf, double error, double terror, int trial)
{
    printf("%d %.5f %.5f %d", trial, error, terror, xcsf->pset.size);
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        printf(" %.5f", set_mean_mut(xcsf, &xcsf->pset, i));
    }
    printf("\n");
    fflush(stdout);
#ifdef GNUPLOT
    gplot_perf2(xcsf, error, terror, trial);
#endif
}
