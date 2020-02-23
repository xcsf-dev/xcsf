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
 * @file neural_activation_kernels.cu
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief CUDA neural network activation functions.
 */ 

#include "neural_activations.h"
#include "cuda.h"

__device__ double logistic_activate_kernel(double x){return 1./(1.+exp(-x));}
__device__ double logistic_gradient_kernel(double x){double fx=1./(1.+exp(-x)); return (1-fx)*fx;}
__device__ double loggy_activate_kernel(double x){return 2./(1.+exp(-x))-1;}
__device__ double loggy_gradient_kernel(double x){double fx=exp(x); return (2.*fx)/pow(fx+1.,2);}
__device__ double gaussian_activate_kernel(double x){return exp(-x*x);}
__device__ double gaussian_gradient_kernel(double x){return -2*x*exp(-x*x);}
__device__ double relu_activate_kernel(double x){return x*(x>0);}
__device__ double relu_gradient_kernel(double x){return (x>0);}
__device__ double linear_activate_kernel(double x){return x;}
__device__ double linear_gradient_kernel(double x){(void)x; return 1;}
__device__ double soft_plus_activate_kernel(double x){return log1p(exp(x));}
__device__ double soft_plus_gradient_kernel(double x){return 1./(1.+exp(-x));}
__device__ double tanh_activate_kernel(double x){return tanh(x);}
__device__ double tanh_gradient_kernel(double x){double t=tanh(x); return 1-t*t;}
__device__ double leaky_activate_kernel(double x){return (x>0) ? x : .1*x;}
__device__ double leaky_gradient_kernel(double x){return (x<0) ? .1 : 1;}
__device__ double sin_activate_kernel(double x){return sin(x);}
__device__ double sin_gradient_kernel(double x){return cos(x);}
__device__ double cos_activate_kernel(double x){return cos(x);}
__device__ double cos_gradient_kernel(double x){return -sin(x);}
__device__ double selu_activate_kernel(double x)
{ return (x>=0)*1.0507*x+(x<0)*1.0507*1.6732*expm1(x); }
__device__ double selu_gradient_kernel(double x)
{ return (x>=0)*1.0507+(x<0)*(1.0507*1.6732*exp(x)); }

__device__ double activate_kernel(int a, double x)
{
    switch(a) {
        case LOGISTIC: return logistic_activate_kernel(x);
        case RELU: return relu_activate_kernel(x);
        case GAUSSIAN: return gaussian_activate_kernel(x);
        case TANH: return tanh_activate_kernel(x);
        case SIN: return sin_activate_kernel(x);
        case COS: return cos_activate_kernel(x);
        case SOFT_PLUS: return soft_plus_activate_kernel(x);
        case LINEAR: return linear_activate_kernel(x);
        case LEAKY: return leaky_activate_kernel(x);
        case SELU: return selu_activate_kernel(x);
        case LOGGY: return loggy_activate_kernel(x);
    }
    return 0;
}

__device__ double gradient_kernel(int a, double x)
{
    switch(a) {
        case LOGISTIC: return logistic_gradient_kernel(x);
        case RELU: return relu_gradient_kernel(x);
        case GAUSSIAN: return gaussian_gradient_kernel(x);
        case TANH: return tanh_gradient_kernel(x);
        case SIN: return sin_gradient_kernel(x);
        case COS: return cos_gradient_kernel(x);
        case SOFT_PLUS: return soft_plus_gradient_kernel(x);
        case LINEAR: return linear_gradient_kernel(x);
        case LEAKY: return leaky_gradient_kernel(x);
        case SELU: return selu_gradient_kernel(x);
        case LOGGY: return loggy_gradient_kernel(x);
    }
    return 0;
}

__global__ void activate_array_kernel(double *state, double *output, int n, int a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
        if(state[i] < NEURON_MIN_STATE) { state[i] = NEURON_MIN_STATE; }
        else if (state[i] > NEURON_MAX_STATE) { state[i] = NEURON_MAX_STATE; }
        output[i] = activate_kernel(a, state[i]);
    }
}

__global__ void gradient_array_kernel(const double *x, int n, int a, double *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
        delta[i] *= gradient_kernel(a, x[i]);
    }
}

extern "C" void activate_array_gpu(double *state, double *output, int n, int a)
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK_SIZE>>>(state, output, n, a);
}

extern "C" void gradient_array_gpu(const double *x, double *delta, int n, int a)
{
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK_SIZE>>>(x, n, a, delta);
}
