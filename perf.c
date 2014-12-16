/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include "cons.h"
#include "perf.h"
#include "cl.h"
#include "cl_set.h"
  
FILE *fout;
char fname[30];
char basefname[30];
 
#ifdef GNUPLOT
void gplot_init();
void gplot_draw();
void gplot_close();
FILE *gp;
#endif

void disp_perf(double *error, int trial, int pnum)
{
	double serr = 0.0;
	for(int i = 0; i < PERF_AVG_TRIALS; i++)
		serr += error[i];
	serr /= (double)PERF_AVG_TRIALS;
	printf("%d %.5f %d", trial, serr, pnum);
	fprintf(fout, "%d %.5f %d", trial, serr, pnum);
#ifdef SELF_ADAPT_MUTATION
	for(int i = 0; i < NUM_MU; i++) {
		printf(" %.5f", avg_mut(&pset, i));
		fprintf(fout, " %.5f", avg_mut(&pset, i));
	}
#endif
	printf("\n");
	fprintf(fout, "\n");
	fflush(stdout);
	fflush(fout);
#ifdef GNUPLOT
	gplot_draw();
#endif
}          
void gen_outfname()
{
	// file for writing output; uses the date/time/exp as file name
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	sprintf(basefname, "out/%04d-%02d-%02d-%02d%02d%02d", tm.tm_year + 1900, 
			tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

void outfile_init(int exp_num)
{                	
	// create output file
	sprintf(fname, "%s-%d.dat", basefname, exp_num);
	fout = fopen(fname, "wt");
	if(fout == 0) {
		printf("Error opening file: %s. %s.\n", fname, strerror(errno));
		exit(EXIT_FAILURE);
	}       
#ifdef GNUPLOT
	gplot_init();
#endif
}

void outfile_close()
{
	fclose(fout);
#ifdef GNUPLOT
	gplot_close();
#endif     
}

#ifdef GNUPLOT
void gplot_init()
{
#ifdef _WIN32
	gp = _popen("C:\Program Files (x86)\gnuplot\bin\pgnuplot.exe -persistent", "w");
#else
	gp = popen("gnuplot -persistent", "w");
#endif
	if(gp != NULL) {
		fprintf(gp, "set terminal wxt noraise enhanced font 'Arial,12'\n");
		fprintf(gp, "set grid\n");
		fprintf(gp, "set border linewidth 1\n");
		fprintf(gp, "set nokey\n");
		fprintf(gp, "set xlabel 'trials'\n");
		fprintf(gp, "set ylabel 'system error'\n");
		fprintf(gp, "set style line 1 lt -1 lw 1 ps 1 lc rgb 'red'\n");
		fprintf(gp, "set style line 2 lt -1 lw 1 ps 1 lc rgb 'blue'\n");
	}
	else {
		printf("error starting gnuplot\n");
	}
}

void gplot_close()
{
	if(gp != NULL)
		pclose(gp);
	else
		printf("error closing gnuplot\n");
}

void gplot_draw()
{
	if(gp != NULL) {
		fprintf(gp, "plot '%s' using 1:2 title 'error' w lp ls 1 pt 4 pi 50\n", fname);
		fprintf(gp,"replot\n");
		fflush(gp);
	}
	else {
		printf("error writing to gnuplot\n");
	}
}
#endif    
