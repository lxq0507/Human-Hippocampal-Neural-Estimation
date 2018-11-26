/*
 * lassoGPUKernel, put the most computationally expensive codes into
 * cuda kernel.
 *
 * By dalegebit
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"

#define NUM_BLOCK 13
#define NUM_THREAD 192

#define CUDA_CALL(res, msg) {\
    if((res) != cudaSuccess) {\
        char buf[100];\
        sprintf(buf, "%s %d", msg, __LINE__);\
        mexErrMsgIdAndTxt(cudaErrId, buf);\
    }\
}       
#define CUBLAS_CALL(res, msg) {\
    if((res) != CUBLAS_STATUS_SUCCESS) {\
        char buf[100];\
        sprintf(buf, "%s %d", msg, __LINE__);\
        mexErrMsgIdAndTxt(cublasErrId, buf);\
    }\
}
#define CUSOLVER_CALL(res, msg) {\
    if((res) != CUSOLVER_STATUS_SUCCESS) {\
        char buf[100];\
        sprintf(buf, "%s %d", msg, __LINE__);\
        mexErrMsgIdAndTxt(cusolverErrId, buf);\
    }\
}
#define MAX(a,b) (((a)>(b))?(a):(b))
#define XORSWAP(a, b)   ((&(a) == &(b)) ? (a) : ((a)^=(b),(b)^=(a),(a)^=(b)))

char const * const inputErrId = "parallel:gpu:lassoFitKernel:InvalidInput";
char const * const inputErrMsg = "Invalid input to MEX file.";
char const * const cudaErrId = "parallel:gpu:lassoFitKernel:CUDAError";
char const * const cudaErrMsg = "CUDA error.";
char const * const cublasErrId = "parallel:gpu:lassoFitKernel:CUBLASError";
char const * const cublasErrMsg = "CUBLAS error.";
char const * const cusolverErrId = "parallel:gpu:lassoFitKernel:CUSOLVERError";
char const * const cusolverErrMsg = "CUBLAS error.";

cudaError_t cudaStat;
cublasStatus_t cublasStat;
cublasHandle_t cublasHandle;
cusolverStatus_t cusolverStat;
cusolverDnHandle_t cusolverHandle;
cudaStream_t cusolverStream = NULL;
cudaStream_t memcpyStream = NULL;

dim3 dimGrid(NUM_BLOCK,1,1);
dim3 dimBlock(NUM_THREAD,1,1);

void __global__ shrinkage(float *z, float kappa, int P, int nThreads)
{
    int i, idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (i = idx; i < P; i += nThreads)
        z[i] = MAX(0, z[i]-kappa)-MAX(0, -z[i]-kappa);
}

void __global__ addEye(float *L, float rho, int P, int nThreads)
{
    int i, idx = blockIdx.x*blockDim.x+threadIdx.x;
    for (i = idx; i < P; i += nThreads)
        L[i*P+i] += rho;
}

float getErr(float const *x, float *oldx, int P) {
    float a = -1, sum;
    // oldx = oldx - x
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &a, x, 1, oldx, 1), cublasErrMsg);
    CUBLAS_CALL(cublasSasum(cublasHandle, P, oldx, 1, &sum), cublasErrMsg);
    return sum;    
}

void singleStep(float const *Xtx, float const *Xty, float const *L,
                int N, int P, float lambda, float rho, float relPar,
                float *x, float *z, float *u,
                // float const *X, float const *Xq, // needed when N < P
                int *devInfo) 
{
    float a, b;
    // /* copy u to oldu asynchronisely */
    // CUDA_CALL(cudaMemcpyAsync(  oldu, u, P*sizeof(float), cudaMemcpyDeviceToDevice, 
    //                             memcpyStream), cudaErrMsg);
    /*---------------------------------------
    * x update 
    *   x = (Xtx + rho*I)^-1(Xty + rho(z-u))
    *---------------------------------------*/
    // x = z
    CUDA_CALL(cudaMemcpy(x, z, P*sizeof(float), cudaMemcpyDeviceToDevice), cudaErrMsg); 
    a = -1, b = 1;
    // x = x-u
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &a, u, 1, x, 1), cublasErrMsg);
    // x = rho*x
    CUBLAS_CALL(cublasSscal(cublasHandle, P, &rho, x, 1), cublasErrMsg);
    // x = Xty+x
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &b, Xty, 1, x, 1), cublasErrMsg);
    if (N >= P) {
        // x = (Xtx + rho*I)^{-1}*x, (Xtx + rho*I)=LU 
        CUSOLVER_CALL(cusolverDnSpotrs( cusolverHandle, CUBLAS_FILL_MODE_LOWER,
                                        P, 1, L, P, x, P, devInfo), cusolverErrMsg);
    }
    // else {
    //     // Xq = X*q
    //     // Xq = (Xtx + rho*I)^{-1}Xq
    //     CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &a, u, 1, x, 1), cublasErrMsg);
    //     CUSOLVER_CALL(cusolverDnSpotrs( cusolverHandle, CUBLAS_FILL_MODE_LOWER,
    //         P, 1, L, P, x, devInfo));
    // }
    
    /*--------------------------------------
    * z update with relaxation
    *   z = S_{lambda/rho}(x_h + u)
    *   x_h = relPar*x + (1-relPar)*z
    *
    *               / a-k, a > k    
    *   S_{k}(a) = |  0,   |a| <= k 
    *               \ a+k, a < -k
    *--------------------------------------*/
    // // wait for zold=z
    // CUDA_CALL(cudaStreamSynchronize(memcpyStream), cudaErrMsg);
    a = 1-relPar;
    // z = (1-relPar)*z
    CUBLAS_CALL(cublasSscal(cublasHandle, P, &a, z, 1), cublasErrMsg);
    // wait for x 
    CUDA_CALL(cudaStreamSynchronize(cusolverStream), cudaErrMsg);
    // z = relPar*x+z
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &relPar, x, 1, z, 1), cublasErrMsg);
    // z = z + u
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &b, u, 1, z, 1), cublasErrMsg);
    // z = shrinkage(z)
    shrinkage<<<dimGrid, dimBlock>>>(z, lambda/rho, P, NUM_BLOCK*NUM_THREAD);

    /*--------------------------------------
    * u update
    *   u = u + rho*(x - z)
    *--------------------------------------*/
    // // wait for oldu=u
    // CUDA_CALL(cudaStreamSynchronize(memcpyStream), cudaErrMsg);
    a = -rho;
    // u = u + rho*x
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &rho, x, 1, u, 1), cublasErrMsg);
    // u = u - rho*z
    CUBLAS_CALL(cublasSaxpy(cublasHandle, P, &a, z, 1, u, 1), cublasErrMsg);
}

/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[],
                int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/

    mxGPUArray const *X, *wX, *Y;
    double const *lambda; /* Matlab array is double by default */
    mxGPUArray *B;
    float const *d_X, *d_wX, *d_Y;
    float *d_Xtx, *d_Xty, *d_L, *LWorkSpace, *newx, *oldx, *tmp;
    float *d_B, *d_bufx, *d_initz, *d_initu, *d_z, *d_u;
    float **d_x;
    int *devInfo;
    int N, P, nLambda, maxIter, LWorkSize;
    int n, i, j, k;
    float a, b, lam, lambdaMax, rho, reltol, threshold, muX, sigmaX, 
        shrinkFactor, totalWeight, relPar=1.2;
    bool standardize;

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;

    /* Initialize the MathWorjs GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=13) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))
        || !(mxIsGPUArray(prhs[2]))) {
        mexErrMsgIdAndTxt(inputErrId, inputErrMsg);
    }
    
    /* 
    * By order, the input parameters are: 
    * X, wX, Y, N, P, lambda, nLambda, lambdaMax, 
    * maxIter, rho, standardize, muX, sigmaX
    */
    X = mxGPUCreateFromMxArray(prhs[0]);
    wX = mxGPUCreateFromMxArray(prhs[1]);
    Y = mxGPUCreateFromMxArray(prhs[2]);
    N = (float)*mxGetPr(prhs[3]);
    P = (float)*mxGetPr(prhs[4]);
    lambda = mxGetPr(prhs[5]);
    nLambda = (int)*mxGetPr(prhs[6]);
    lambdaMax = (float)*mxGetPr(prhs[7]);
    maxIter = (int)*mxGetPr(prhs[8]);
    rho = (float)*mxGetPr(prhs[9]);
    standardize = (bool)*mxGetPr(prhs[10]);
    muX = (float)*mxGetPr(prhs[11]);
    sigmaX = (float)*mxGetPr(prhs[12]);

    /*
    * Verify that arrays really are float arrays before extracting the pointer.
    */
    if ((mxGPUGetClassID(X) != mxSINGLE_CLASS) || (mxGPUGetClassID(wX) != mxSINGLE_CLASS) 
        || (mxGPUGetClassID(Y) != mxSINGLE_CLASS)){
        mexErrMsgIdAndTxt(inputErrId, inputErrMsg);
    }

    d_X = (float const *)(mxGPUGetDataReadOnly(X));
    d_wX = (float const *)(mxGPUGetDataReadOnly(wX));
    d_Y = (float const *)(mxGPUGetDataReadOnly(Y));

    CUDA_CALL(cudaMalloc((void**)&d_Xtx, P*P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMalloc((void**)&d_Xty, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMalloc((void**)&d_L, P*P*sizeof(float)), "Cuda out of memory.");

    CUDA_CALL(cudaMalloc((void**)&d_bufx, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMemset(d_bufx, 0, P*sizeof(float)), "Cuda memset error.");
    CUDA_CALL(cudaMalloc((void**)&d_initz, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMemset(d_initz, 0, P*sizeof(float)), "Cuda memset error.");
    CUDA_CALL(cudaMalloc((void**)&d_initu, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMemset(d_initu, 0, P*sizeof(float)), "Cuda memset error.");
    CUDA_CALL(cudaMalloc((void**)&d_z, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMemset(d_z, 0, P*sizeof(float)), "Cuda memset error.");
    CUDA_CALL(cudaMalloc((void**)&d_u, P*sizeof(float)), "Cuda out of memory.");
    CUDA_CALL(cudaMemset(d_u, 0, P*sizeof(float)), "Cuda memset error.");
    CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)), "Cuda out of memory.");


    d_x = (float**) malloc(nLambda*sizeof(float*));
    for (i = 0; i < nLambda; ++i) {
        CUDA_CALL(cudaMalloc((void**)&(d_x[i]), P*sizeof(float)), "Cuda out of memory.");
    }

    CUBLAS_CALL(cublasCreate(&cublasHandle), 
                "Cublas init fail.");
    CUSOLVER_CALL(cusolverDnCreate(&cusolverHandle), 
                "Cusolver init fail.");
    CUDA_CALL(cudaStreamCreateWithFlags(&cusolverStream, cudaStreamNonBlocking), 
                "Cuda cusolverStream init fail.");
    CUSOLVER_CALL(cusolverDnSetStream(cusolverHandle, cusolverStream), 
                "Set cusolverStream fail.");
    CUDA_CALL(cudaStreamCreateWithFlags(&memcpyStream, cudaStreamNonBlocking), 
                "Cuda memcpyStream init fail.");

    totalWeight = N;
    a = 1, b = 0;
    // Xtx = Xt * X
    CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, P, N, P, &a, d_X, N, 
                            d_X, N, &b, d_Xtx, P), cublasErrMsg);
    CUDA_CALL(cudaMemcpy(d_L, d_Xtx, P*P*sizeof(float), cudaMemcpyDeviceToDevice), cudaErrMsg);
    // Xty = Xt * y
    CUBLAS_CALL(cublasSgemv(cublasHandle, CUBLAS_OP_T, N, P, &a, d_X, N, d_Y, 1, 
                            &b, d_Xty, 1), cublasErrMsg);
    // Xtx+rho*I = L*Lt
    CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(  cusolverHandle, CUBLAS_FILL_MODE_LOWER,
                                                P, d_L, P, &LWorkSize), cusolverErrMsg);

    CUDA_CALL(cudaMalloc((void**)&LWorkSpace, LWorkSize*sizeof(float)), cudaErrMsg);
    addEye<<<dimGrid, dimBlock>>>(d_L, rho, P, NUM_THREAD*NUM_BLOCK);
    CUSOLVER_CALL(cusolverDnSpotrf( cusolverHandle, CUBLAS_FILL_MODE_LOWER,
                                    P, d_L, P, LWorkSpace, LWorkSize, devInfo), cusolverErrMsg);
    cudaFree(LWorkSpace);

    for (n = 0; n < nLambda; ++n) {
        lam = (float)lambda[n];
        if (lam >= lambdaMax)
            continue;
        cudaMemcpy(d_z, d_initz, P*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_u, d_initu, P*sizeof(float), cudaMemcpyDeviceToDevice);
        newx = d_x[n];
        oldx = d_bufx;
        for (i = 0; i < maxIter; ++i) {
            singleStep(d_Xtx, d_Xty, d_L, N, P, lam, rho, relPar, newx, d_z, d_u, devInfo);
            if (i % 10 == 0 && getErr(newx, oldx, P) < reltol)
                break; 
            // XORSWAP(newx, oldx);
            tmp = newx;
            newx = oldx;
            oldx = tmp;
        } // End i
        if (newx != d_x[n])
            cudaMemcpy(d_x[n], d_bufx, P*sizeof(float), cudaMemcpyDeviceToDevice);
    } // End n

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    mwSize BDims[2] = {P, nLambda};
    B = mxGPUCreateGPUArray(2, // nDim
                            BDims, 
                            mxSINGLE_CLASS,
                            mxREAL,
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));
    for (i = 0; i < nLambda; ++i) {
        cudaMemcpy(&d_B[i*P], d_x[i], P*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
    * The mxGPUArray pointers are host-side structures that refer to device
    * data. These must be destroyed before leaving the MEX function.
    */
    mxGPUDestroyGPUArray(B);

    cudaFree(d_Xtx), cudaFree(d_Xty), cudaFree(d_L);
    cudaFree(d_bufx), cudaFree(d_initz), cudaFree(d_initu);
    cudaFree(d_z), cudaFree(d_u);
    for (i = 0; i < nLambda; ++i)
        cudaFree(d_x[i]);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    free(d_x);
}
