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
 * The performance output module.
 *
 * Writes system performance to a file and standard out. If GNUPlot is enabled,
 * a 2D plot is redrawn each time the performance is updated.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include "data_structures.h"
#include "cl.h"
#include "cl_set.h"
#include "perf.h"

#ifdef GNUPLOT
FILE *fout;
char fname[30];
char basefname[30];

void gplot_init(XCSF *xcsf);
void gplot_draw(XCSF *xcsf);
void gplot_close(XCSF *xcsf);
FILE *gp;
#endif

void disp_perf(XCSF *xcsf, double *error, double *terror, int trial)
{
	double serr = 0.0;
	double terr = 0.0;
	for(int i = 0; i < xcsf->PERF_AVG_TRIALS; i++) {
		serr += error[i];
		terr += terror[i];
	}
	serr /= (double)xcsf->PERF_AVG_TRIALS;
	terr /= (double)xcsf->PERF_AVG_TRIALS;
	printf("%d %.5f %.5f %d", trial, serr, terr, xcsf->pop_num);
	for(int i = 0; i < xcsf->NUM_SAM; i++) {
		printf(" %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
	}
	printf("\n");    
	fflush(stdout);

#ifdef GNUPLOT
	fprintf(fout, "%d %.5f %.5f %d", trial, serr, terr, xcsf->pop_num);
	for(int i = 0; i < xcsf->NUM_SAM; i++) {
		fprintf(fout, " %.5f", set_avg_mut(xcsf, &xcsf->pset, i));
	}
	fprintf(fout, "\n");
	fflush(fout);

	gplot_draw(xcsf);
#endif
}          

#ifdef GNUPLOT
void gen_outfname(XCSF *xcsf)
{
	// file for writing output; uses the date/time/exp as file name
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	sprintf(basefname, "out/%04d-%02d-%02d-%02d%02d%02d", tm.tm_year + 1900, 
			tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	(void)xcsf;
}

void outfile_init(XCSF *xcsf, int exp_num)
{                	
	// create output file
	sprintf(fname, "%s-%d.dat", basefname, exp_num);
	fout = fopen(fname, "wt");
	if(fout == 0) {
		printf("Error opening file: %s. %s.\n", fname, strerror(errno));
		exit(EXIT_FAILURE);
	}       
	gplot_init(xcsf);
}

void outfile_close(XCSF *xcsf)
{
	fclose(fout);
	gplot_close(xcsf);
}

void gplot_init(XCSF *xcsf)
{
	// set gnuplot title
	char buffer[20];
	char title[200];
	title[0] = '\0';

	switch(xcsf->COND_TYPE) {
		case -1:
			strcat(title, " DUMMY COND");
			break;
		case 0:
			strcat(title, " RECT COND");
			break;
		case 1:
			strcat(title, " NEURAL COND");
			break;
		case 2:
			strcat(title, " TREE-GP COND");
			break;
		case 3:
			strcat(title, " GRAPH-DGP COND");
			break;
		case 11:
			strcat(title, " GRAPH-DGP RULE");
			break;
		case 12:
			strcat(title, " NEURAL RULE");
			break;
	}

	if(xcsf->COND_TYPE < 10) {
		switch(xcsf->PRED_TYPE) {
			case 0:
				strcat(title, ", LINEAR NLMS");
				break;
			case 1:
				strcat(title, ", QUADRATIC NLMS");
				break;
			case 2:
				strcat(title, ", LINEAR RLS");
				break;
			case 3:
				strcat(title, ", QUADRATIC RLS");
				break;
			case 4:
				strcat(title, ", NEURAL PRED");
				break;
		}
	}

	if(xcsf->NUM_SAM > 0) {
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

void gplot_close(XCSF *xcsf)
{
	if(gp != NULL) {
		pclose(gp);
	}
	else {
		printf("error closing gnuplot\n");
	}
	(void)xcsf;
}

void gplot_draw(XCSF *xcsf)
{
	if(gp != NULL) {
		fprintf(gp, "plot '%s' using 1:2 title 'train error' w lp ls 1 pt 4 pi 50, ", fname);
		fprintf(gp, "'%s' using 1:3 title 'test error' w lp ls 2 pt 8 pi 50\n", fname);
		fprintf(gp,"replot\n");
		fflush(gp);
	}
	else {
		printf("error writing to gnuplot\n");
	}
	(void)xcsf;
}
#endif    
