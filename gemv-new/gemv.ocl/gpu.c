/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"

void compute_gpu(dataset_t* gpu, struct pb_Parameters *parameters) {
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found.");
    return;
  }

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"");  //-cl-nv-verbose

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  if (clStatus != CL_SUCCESS) {
    size_t string_size = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
                          0, NULL, &string_size);
    char* string = malloc(string_size*sizeof(char));
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
                          string_size, string, NULL);
    puts(string);
    free(string);
  }

  {
    // Query binary (PTX file) size
    size_t bin_sz;
    int err;
    err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

    // Read binary (PTX file) to memory buffer
    unsigned char *bin = (unsigned char *)malloc(bin_sz);
    err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);

    // Save PTX to add_vectors_ocl.ptx
    FILE* fp = fopen("sgemv.ptx", "wb");
    fwrite(bin, sizeof(char), bin_sz, fp);
    fclose(fp);
    free(bin);
  }

  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"sgemv",&clStatus);
  CHECK_ERROR("clCreateKernel")

  int nRows = gpu->nRows;
  int nCols = gpu->nCols;
  float* A = gpu->A;
  float* x = gpu->x;
  float* y = gpu->y;
  float alpha = gpu->alpha;
  float beta  = gpu->beta ;

  cl_mem A_d;
  cl_mem x_d;
  cl_mem y_d;
  A_d  = clCreateBuffer(clContext,CL_MEM_READ_ONLY,sizeof(float)*nRows*nCols,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  x_d  = clCreateBuffer(clContext,CL_MEM_READ_ONLY,sizeof(float)*nCols,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  y_d = clCreateBuffer(clContext,CL_MEM_READ_WRITE,sizeof(float)*nRows,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  clStatus = clEnqueueWriteBuffer(clCommandQueue,A_d,CL_TRUE,0,sizeof(float)*nRows*nCols,A,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,x_d,CL_TRUE,0,sizeof(float)*nCols,x,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,y_d,CL_TRUE,0,sizeof(float)*nRows,y,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  clStatus  = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &y_d);
  clStatus |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &A_d);
  clStatus |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &x_d);
  clStatus |= clSetKernelArg(clKernel, 3, sizeof(float),  &alpha);
  clStatus |= clSetKernelArg(clKernel, 4, sizeof(float),  &beta);
  clStatus |= clSetKernelArg(clKernel, 5, sizeof(int),    &nRows);
  clStatus |= clSetKernelArg(clKernel, 6, sizeof(int),    &nCols);
  CHECK_ERROR("clSetKernelArg")

  /* loop over z-dimension, invoke OpenCL kernel for each x-y plane */
  size_t blockDim[1] = { 128 };
  size_t gridDim[1] = { ((nRows + blockDim[0] - 1) / blockDim[0]) * blockDim[0] };
  {
    double start, end;
    clFinish(clCommandQueue);
    start = get_sec();
    clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,1,NULL,gridDim,blockDim,0,NULL,NULL);
    clFinish(clCommandQueue);
    CHECK_ERROR("clEnqueueNDRangeKernel")
    end = get_sec();
    printf(" %6dx%d : ", (int)nRows,(int)nCols);
    printf(" %10.2f MFlops\n", 2. * (double)nRows * (double)nCols / (end-start) * 1.e-6);
  }
  clStatus = clFinish(clCommandQueue);
  CHECK_ERROR("clFinish")

  clStatus = clEnqueueReadBuffer(clCommandQueue,y_d,CL_TRUE,0,sizeof(float)*nRows,y,0,NULL,NULL);
  CHECK_ERROR("clEnqueueReadBuffer")

  /* free OpenCL memory allocations */
  clStatus = clReleaseMemObject(A_d);
  clStatus = clReleaseMemObject(x_d);
  clStatus = clReleaseMemObject(y_d);
  CHECK_ERROR("clReleaseMemObject")

  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext);

  free((void*)clSource[0]);
}

