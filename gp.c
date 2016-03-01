/*
 * Copyright (C) 2016 Riintd Preen <rpreen@gmail.com>
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
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "random.h"
#include "cons.h"
#include "gp.h"

int tree_grow(int *buffer, int p, int max, int depth);
int tree_traverse(int *tree, int p);

void tree_init_cons()
{
	for(int i = 0; i < GP_CONS; i++)
		cons[i] = (2.0*drand())-1.0;
}     

void tree_init(GP_TREE *gp)
{
	gp->tree = malloc(sizeof(int)*1);
}

void tree_rand(GP_TREE *gp)
{
	// create new random tree
	int buffer[GP_MAX_LEN];
	int len = 0;
	do {
		len = tree_grow(buffer, 0, GP_MAX_LEN, GP_DEPTH);
	} while(len < 0);

	// copy tree to this individual
	gp->tree = malloc(sizeof(int)*len);
	memcpy(gp->tree, buffer, sizeof(int)*len);
}

void tree_free(GP_TREE *gp)
{
	free(gp->tree);
}
 
int tree_grow(int *buffer, int p, int max, int depth)
{
	int prim = irand(0,2);
	int one_child;

	if(p >= max) 
		return(-1);
	if(p == 0)
		prim = 1;

	// add constant or external input
	if(prim == 0 || depth == 0) {
		prim = irand(GP_NUM_FUNC, GP_NUM_FUNC+GP_CONS+state_length);
		buffer[p] = prim;
		return(p+1);
	}
	// add new function
	else {
		prim = irand(0,GP_NUM_FUNC);
		switch(prim) {
			case ADD: 
			case SUB: 
			case MUL: 
			case DIV:
				buffer[p] = prim;
				one_child = tree_grow(buffer, p+1, max, depth-1);
				if(one_child < 0) 
					return(-1);
				return(tree_grow(buffer, one_child, max, depth-1));
		}
	}
	printf("grow() shouldn't be here\n");
	return(0);
}

double tree_eval(GP_TREE *gp, double *state)
{
	int node = gp->tree[(gp->p)++];

	// external input
	if(node >= GP_NUM_FUNC+GP_CONS)
		return(state[node-GP_NUM_FUNC-GP_CONS]);
	// constant
	else if(node >= GP_NUM_FUNC)
		return(cons[node-GP_NUM_FUNC]);

	// function
	switch(node) {
		case ADD : return(tree_eval(gp,state) + tree_eval(gp,state));
		case SUB : return(tree_eval(gp,state) - tree_eval(gp,state));
		case MUL : return(tree_eval(gp,state) * tree_eval(gp,state));
		case DIV : { 
					   double num = tree_eval(gp,state); 
					   double den = tree_eval(gp,state);
					   if(den == 0.0) 
						   return(num);
					   else 
						   return(num/den);
				   }
	}
	printf("eval() shouldn't be here\n");
	return(0.0);
}

int tree_print(GP_TREE *gp, int p) 
{
	int node = gp->tree[p];

	if(node >= GP_NUM_FUNC) {
		// external input
		if(node >= GP_NUM_FUNC+GP_CONS)
			printf("IN:%d ", node-GP_NUM_FUNC-GP_CONS);
		// constant
		else
			printf("%f", cons[node-GP_NUM_FUNC]);
		return(++p);
	}

	// function
	int a1 = 0; 
	int a2 = 0;
	switch(node) {
		case ADD: printf( "(");
				  a1 = tree_print(gp, ++p); 
				  printf( " + "); 
				  break;
		case SUB: printf( "(");
				  a1 = tree_print(gp, ++p); 
				  printf( " - "); 
				  break;
		case MUL: printf( "(");
				  a1 = tree_print(gp, ++p); 
				  printf( " * "); 
				  break;
		case DIV: printf( "(");
				  a1 = tree_print(gp, ++p); 
				  printf( " / "); 
				  break;
	}
	a2 = tree_print(gp, a1); 
	printf(")"); 
	return(a2);
}

void tree_copy(GP_TREE *to, GP_TREE *from)
{
	free(to->tree);
	int len = tree_traverse(from->tree, 0);
	to->tree = malloc(sizeof(int)*len);
	memcpy(to->tree, from->tree, sizeof(int)*len);
	to->p = from->p;               
}

void tree_crossover(GP_TREE *p1, GP_TREE *p2)
{
	int len1 = tree_traverse(p1->tree, 0);
	int len2 = tree_traverse(p2->tree, 0);
	int start1 = irand(0,len1);
	int end1 = tree_traverse(p1->tree, start1);
	int start2 = irand(0,len2);
	int end2 = tree_traverse(p2->tree, start2);

	int nlen1 = start1+(end2-start2)+(len1-end1);
	int *new1 = malloc(sizeof(int)*nlen1);
	memcpy(&new1[0], &p1->tree[0], sizeof(int)*start1);
	memcpy(&new1[start1], &p2->tree[start2], sizeof(int)*(end2-start2));
	memcpy(&new1[start1+(end2-start2)], &p1->tree[end1], sizeof(int)*(len1-end1));

	int nlen2 = start2+(end1-start1)+(len2-end2);
	int *new2 = malloc(sizeof(int)*nlen2);
	memcpy(&new2[0], &p2->tree[0], sizeof(int)*start2);
	memcpy(&new2[start2], &p1->tree[start1], sizeof(int)*(end1-start1));
	memcpy(&new2[start2+(end1-start1)], &p2->tree[end2], sizeof(int)*(len2-end2));

	tree_free(p1);
	tree_free(p2);
	p1->tree = new1;
	p2->tree = new2;
}

void tree_mutation(GP_TREE *offspring, double rate) 
{   
	int len = tree_traverse(offspring->tree, 0);
	for(int i = 0; i < len; i++) {  
		if(drand() < rate) {
			if(offspring->tree[i] >= GP_NUM_FUNC)
				offspring->tree[i] = irand(GP_NUM_FUNC, GP_NUM_FUNC+GP_CONS+state_length);
			else
				switch(offspring->tree[i]) {
					case ADD: 
					case SUB: 
					case MUL: 
					case DIV:
						offspring->tree[i] = irand(0, GP_NUM_FUNC);
				}
		}
	}
}

int tree_traverse(int *tree, int p)
{
	if(tree[p] >= GP_NUM_FUNC)
		return(++p);

	switch(tree[p]) {
		case ADD: 
		case SUB: 
		case MUL: 
		case DIV: 
			return(tree_traverse(tree, tree_traverse(tree, ++p)));
	}
	printf("traverse() shouldn't be here\n");
	return(0);
}
