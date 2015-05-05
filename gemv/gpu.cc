/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

// from SDK example
#include <helper_cuda.h>

#include "common.h"


/*
 * This is copied from ptxjit
 */
void ptxJIT(CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState)
{
    CUjit_option options[6];
    void* optionVals[6];
    float walltime;
    char error_log[8192],
         info_log[8192];
    // unsigned int logSize = 8192;
    void* logSize = (void*) 8192;
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void*) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void*) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void*) logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void*) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void*) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void*) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6,options, optionVals, lState));

    {
        // Load the PTX from the string myPtx (64-bit)
        printf("Loading myPtx[] program\n");
        // PTX May also be loaded from file, as per below.
        myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "myPtx.ptx",0,0,0);
    }

    if ( myErr != CUDA_SUCCESS )
    {
      // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
      fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n",walltime,info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(phModule, cuOut));

    // Locate the kernel entry point
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "sgemv"));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
}

void compute_gpu(dataset_t* gpu) {

  int cuda_device = 0;
  cuda_device = gpuDeviceInit(cuda_device);
  if (cuda_device < 0) {
    printf ("No CUDA device found.\n");
    exit(EXIT_FAILURE);
  }

  // allocate memory
  float* d_A;
  float* d_x;
  float* d_y;
  float alpha = gpu->alpha;
  float beta = gpu->beta;
  int nRows = gpu->nRows;
  int nCols = gpu->nCols;

  int sz_A = sizeof(float) * nRows * nCols;
  int sz_x = sizeof(float) * nCols;
  int sz_y = sizeof(float) * nRows;

  checkCudaErrors(cudaMalloc(&d_A, sz_A));
  checkCudaErrors(cudaMemcpy(d_A, gpu->A, sz_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_x, sz_x));
  checkCudaErrors(cudaMemcpy(d_x, gpu->x, sz_x, cudaMemcpyHostToDevice));

  // JIT compile the kernel from PTX
  CUmodule     hModule  = 0;
  CUfunction   hKernel  = 0;
  CUlinkState  lState;
  ptxJIT(&hModule, &hKernel, &lState);

  // Set the kernel parameters
  /*
    clStatus  = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &y_d);
    clStatus |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &A_d);
    clStatus |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &x_d);
    clStatus |= clSetKernelArg(clKernel, 3, sizeof(float),  &alpha);
    clStatus |= clSetKernelArg(clKernel, 4, sizeof(float),  &beta);
    clStatus |= clSetKernelArg(clKernel, 5, sizeof(int),    &nRows);
    clStatus |= clSetKernelArg(clKernel, 6, sizeof(int),    &nCols);
  */
  int nThreads = 32;
  int nBlocks = nRows / nThreads;

  int paramOffset = 0;
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_y, sizeof(d_y)));
  paramOffset += sizeof(d_y);
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_A, sizeof(d_A)));
  paramOffset += sizeof(d_A);
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_x, sizeof(d_x)));
  paramOffset += sizeof(d_x);
#if 0
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &alpha, sizeof(&alpha)));
  paramOffset += sizeof(&alpha);
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &beta, sizeof(&beta)));
  paramOffset += sizeof(&beta);
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &nRows, sizeof(&nRows)));
  paramOffset += sizeof(&nRows);
  checkCudaErrors(cuParamSetv(hKernel, paramOffset, &nCols, sizeof(&nCols)));
  paramOffset += sizeof(&nCols);
#else
  checkCudaErrors(cuParamSetf(hKernel, paramOffset, alpha));
  paramOffset += sizeof(float);
  checkCudaErrors(cuParamSetf(hKernel, paramOffset, beta));
  paramOffset += sizeof(float);
  checkCudaErrors(cuParamSeti(hKernel, paramOffset, nRows));
  paramOffset += sizeof(int);
  checkCudaErrors(cuParamSeti(hKernel, paramOffset, nCols));
  paramOffset += sizeof(int);
#endif
  checkCudaErrors(cuParamSetSize(hKernel, paramOffset));

  // Launch the kernel (Driver API_)
  fprintf (stderr, "%d,%d,%d\n", nBlocks, nThreads, nRows);
  checkCudaErrors(cuFuncSetBlockShape(hKernel, nThreads, 1, 1));
  checkCudaErrors(cuLaunchGrid(hKernel, nBlocks, 1));
  fprintf (stderr, "CUDA kernel launched!\n");

  checkCudaErrors(cudaMemcpy(gpu->y, d_y, sz_y, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  if (hModule) {
    checkCudaErrors(cuModuleUnload(hModule));
    hModule = 0;
  }
}

