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
#ifdef GNUPLOT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gplot.h"

FILE *gp;
char *datafile;

void gplot_init(char *infile)
{
	datafile = infile;
#ifdef _WIN32
	gp = _popen("C:\Program Files (x86)\gnuplot\bin\pgnuplot.exe", "w");
#else
	gp = popen("gnuplot", "w");
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
		fprintf(gp, "plot '%s' using 1:2 title 'error' w lp ls 1 pt 4 pi 50\n", datafile);
		fprintf(gp,"replot\n");
		fflush(gp);
	}
	else {
		printf("error writing to gnuplot\n");
	}
}
#endif    
