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
 * @file pred_rls.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Recursive least mean squares prediction functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "cl.h"
#include "prediction.h"
#include "pred_rls.h"

#ifdef GPU
#include "cuda.h"
#include "blas_kernels.h"
#endif

static void init_matrix(const XCSF *xcsf, double *matrix, int n);
                    
/**
 * @brief Recursive least mean squares prediction data structure.
 */ 
typedef struct PRED_RLS {
    int n; //!< Number of weights for each predicted variable
    int n_weights; //!< Total number of weights
    double *weights; //!< Weights used to compute prediction
    double *matrix; //!< Gain matrix used to update weights
    double *tmp_input; //!< Temporary storage for updating weights
    double *tmp_vec; //!< Temporary storage for updating weights
    double *tmp_matrix1; //!< Temporary storage for updating gain matrix
    double *tmp_matrix2; //!< Temporary storage for updating gain matrix
#ifdef GPU
    cudaStream_t stream;
    double *matrix_gpu;
    double *tmp_input_gpu;
    double *tmp_vec_gpu;
    double *tmp_matrix1_gpu;
    double *tmp_matrix2_gpu;
    double *tmp_gpu;
#endif
} PRED_RLS;

void pred_rls_init(const XCSF *xcsf, CL *c)
{
    PRED_RLS *pred = malloc(sizeof(PRED_RLS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->n = 1 + 2 * xcsf->x_dim + xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    }
    else {
        pred->n = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->n_weights = pred->n * xcsf->y_dim;
    pred->weights = calloc(pred->n_weights, sizeof(double));
    blas_fill(xcsf->y_dim, xcsf->PRED_X0, pred->weights, pred->n);
    // initialise gain matrix
    int n_sqrd = pred->n * pred->n;
    pred->matrix = malloc(n_sqrd * sizeof(double));
    init_matrix(xcsf, pred->matrix, pred->n);
    // initialise temporary storage for weight updating
    pred->tmp_input = calloc(pred->n, sizeof(double));
    pred->tmp_vec = calloc(pred->n, sizeof(double));
    pred->tmp_matrix1 = calloc(n_sqrd, sizeof(double));
    pred->tmp_matrix2 = calloc(n_sqrd, sizeof(double));
#ifdef GPU
    CUDA_CALL( cudaStreamCreate(&pred->stream) );
    pred->matrix_gpu = cuda_make_array(pred->matrix, n_sqrd, &pred->stream);
    pred->tmp_matrix1_gpu = cuda_make_array(pred->tmp_matrix1, n_sqrd, &pred->stream);
    pred->tmp_matrix2_gpu = cuda_make_array(pred->tmp_matrix2, n_sqrd, &pred->stream);
    pred->tmp_input_gpu = cuda_make_array(pred->tmp_input, pred->n, &pred->stream);
    pred->tmp_vec_gpu = cuda_make_array(pred->tmp_vec, pred->n, &pred->stream);
    pred->tmp_gpu = cuda_make_array(NULL, 1, &pred->stream);
#endif
}

static void init_matrix(const XCSF *xcsf, double *matrix, int n)
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i != j) {
                matrix[i*n+j] = 0;
            }
            else {
                matrix[i*n+j] = xcsf->PRED_RLS_SCALE_FACTOR;
            }
        }
    }
}

void pred_rls_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    pred_rls_init(xcsf, to);
    const PRED_RLS *to_pred = to->pred;
    const PRED_RLS *from_pred = from->pred;
    memcpy(to_pred->weights, from_pred->weights, from_pred->n_weights * sizeof(double));
}

void pred_rls_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    PRED_RLS *pred = c->pred;
    free(pred->weights);
    free(pred->matrix);
    free(pred->tmp_input);
    free(pred->tmp_vec);
    free(pred->tmp_matrix1);
    free(pred->tmp_matrix2);
#ifdef GPU
    CUDA_CALL( cudaFree(pred->matrix_gpu) );
    CUDA_CALL( cudaFree(pred->tmp_matrix1_gpu) );
    CUDA_CALL( cudaFree(pred->tmp_matrix2_gpu) );
    CUDA_CALL( cudaFree(pred->tmp_input_gpu) );
    CUDA_CALL( cudaFree(pred->tmp_vec_gpu) );
    CUDA_CALL( cudaFree(pred->tmp_gpu) );
    CUDA_CALL( cudaStreamDestroy(pred->stream) );
#endif
    free(pred);
}

void pred_rls_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    const PRED_RLS *pred = c->pred;
    int n = pred->n;
    pred->tmp_input[0] = xcsf->PRED_X0;
    int idx = 1;
    // linear terms
    for(int i = 0; i < xcsf->x_dim; i++) {
        pred->tmp_input[idx++] = x[i];
    }
    // quadratic terms
    if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        for(int i = 0; i < xcsf->x_dim; i++) {
            for(int j = i; j < xcsf->x_dim; j++) {
                pred->tmp_input[idx++] = x[i] * x[j];
            }
        }
    }
#ifdef GPU
    int n_sqrd = n * n;
    double divisor;
    CUDA_CALL( cudaMemcpyAsync(pred->matrix_gpu, pred->matrix, sizeof(double)*n_sqrd,
                cudaMemcpyHostToDevice, pred->stream) );
    CUDA_CALL( cudaMemcpyAsync(pred->tmp_input_gpu, pred->tmp_input, sizeof(double)*n,
                cudaMemcpyHostToDevice, pred->stream) );

    gemm_gpu(0,0,n,1,n,1,pred->matrix_gpu,n,pred->tmp_input_gpu,1,0,pred->tmp_vec_gpu,1,&pred->stream);
    dot_gpu(n, pred->tmp_input_gpu, 1, pred->tmp_vec_gpu, 1, pred->tmp_gpu, &pred->stream);
    CUDA_CALL( cudaMemcpyAsync(&divisor, pred->tmp_gpu, sizeof(double),
                cudaMemcpyDeviceToHost, pred->stream) );
    divisor = 1 / (divisor + xcsf->PRED_RLS_LAMBDA);
    scal_gpu(n, divisor, pred->tmp_vec_gpu, 1, &pred->stream);
    CUDA_CALL( cudaMemcpyAsync(pred->tmp_vec, pred->tmp_vec_gpu, sizeof(double)*n,
                cudaMemcpyDeviceToHost, pred->stream) );
#else
    // gain vector = matrix * tmp_input
    blas_gemm(0, 0, n, 1, n, 1, pred->matrix, n, pred->tmp_input, 1, 0, pred->tmp_vec, 1);
    // divide gain vector by lambda + tmp_vec
    double divisor = blas_dot(n, pred->tmp_input, 1, pred->tmp_vec, 1);
    divisor = 1 / (divisor + xcsf->PRED_RLS_LAMBDA);
    blas_scal(n, divisor, pred->tmp_vec, 1);
#endif
    // update weights using the error
    for(int var = 0; var < xcsf->y_dim; var++) {
        double error = y[var] - c->prediction[var];
        blas_axpy(n, error, pred->tmp_vec, 1, &pred->weights[var*n], 1);
    }
    // update gain matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            double tmp = pred->tmp_vec[i] * pred->tmp_input[j];
            if(i == j) {
                pred->tmp_matrix1[i*n+j] = 1 - tmp;
            }
            else {
                pred->tmp_matrix1[i*n+j] = -tmp;
            }
        }
    }
    // tmp_matrix2 = tmp_matrix1 * pred_matrix
#ifdef GPU
    CUDA_CALL( cudaMemcpyAsync(pred->tmp_matrix1_gpu, pred->tmp_matrix1, sizeof(double)*n_sqrd,
                cudaMemcpyHostToDevice, pred->stream) );
    gemm_gpu(0,0,n,n,n,1,pred->tmp_matrix1_gpu,n,pred->matrix_gpu,n,0,pred->tmp_matrix2_gpu,n,&pred->stream);
    CUDA_CALL( cudaMemcpyAsync(pred->tmp_matrix2, pred->tmp_matrix2_gpu, sizeof(double)*n_sqrd,
                cudaMemcpyDeviceToHost, pred->stream) );
#else
    blas_gemm(0, 0, n, n, n, 1, pred->tmp_matrix1, n, pred->matrix, n, 0, pred->tmp_matrix2, n);
#endif
    // divide gain matrix entries by lambda
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            pred->matrix[i*n+j] = pred->tmp_matrix2[i*n+j] / xcsf->PRED_RLS_LAMBDA;
        }
    }
}

void pred_rls_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    const PRED_RLS *pred = c->pred;
    int n = pred->n;
    for(int var = 0; var < xcsf->y_dim; var++) {
        // first coefficient is offset
        double pre = xcsf->PRED_X0 * pred->weights[var*n];
        int idx = 1;
        // multiply linear coefficients with the prediction input
        for(int i = 0; i < xcsf->x_dim; i++, idx++) {
            pre += pred->weights[var*n+idx] * x[i];
        }
        if(xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
            // multiply quadratic coefficients with prediction input
            for(int i = 0; i < xcsf->x_dim; i++) {
                for(int j = i; j < xcsf->x_dim; j++, idx++) {
                    pre += pred->weights[var*n+idx] * x[i] * x[j];
                }
            }
        }
        c->prediction[var] = pre;
    }
} 

void pred_rls_print(const XCSF *xcsf, const CL *c)
{
    const PRED_RLS *pred = c->pred;
    printf("RLS weights: ");
    int n = pred->n;
    for(int var = 0; var < xcsf->y_dim; var++) {
        for(int i = 0; i < n; i++) {
            printf("%f, ", pred->weights[var*n+i]);
        }
        printf("\n");
    }
}

_Bool pred_rls_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool pred_rls_mutate(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

int pred_rls_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const PRED_RLS *pred = c->pred;
    return pred->n_weights;
}

size_t pred_rls_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    const PRED_RLS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->n, sizeof(int), 1, fp);
    s += fwrite(&pred->n_weights, sizeof(int), 1, fp);
    s += fwrite(pred->weights, sizeof(double), pred->n_weights, fp);
    int n_sqrd = pred->n * pred->n;
    s += fwrite(pred->matrix, sizeof(double), n_sqrd, fp);
    return s;
}

size_t pred_rls_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    pred_rls_init(xcsf, c);
    PRED_RLS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->n, sizeof(int), 1, fp);
    s += fread(&pred->n_weights, sizeof(int), 1, fp);
    s += fread(pred->weights, sizeof(double), pred->n_weights, fp);
    int n_sqrd = pred->n * pred->n;
    s += fread(pred->matrix, sizeof(double), n_sqrd, fp);
    return s;
}
