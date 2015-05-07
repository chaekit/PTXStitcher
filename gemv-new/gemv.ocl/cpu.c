#include <stdio.h>
#include <stdlib.h>
#include "common.h"

void create_data(dataset_t* cpu, dataset_t* gpu, int nRows, int nCols) {
  cpu->nRows = nRows;
  cpu->nCols = nCols;
  cpu->alpha = 1.0f;
  cpu->beta = 0.0f;
  cpu->A = (float *) malloc(sizeof(float) * nRows * nCols);
  cpu->x = (float *) malloc(sizeof(float) * nCols);
  cpu->y = (float *) malloc(sizeof(float) * nRows);

  gpu->nRows = nRows;
  gpu->nCols = nCols;
  gpu->alpha = 1.0f;
  gpu->beta = 0.0f;
  gpu->A = cpu->A; // (float *) malloc(sizeof(float) * nRows * nCols);
  gpu->x = cpu->x; // (float *) malloc(sizeof(float) * nCols);
  gpu->y = (float *) malloc(sizeof(float) * nRows);

  int i, j;
  for (i = 0; i < nRows; i++) {
    for (j = 0; j < nCols; j++) {
      cpu->A[i*nRows+j] = rand() % 16 - 8;
    }
  }
  for (i = 0; i < nCols; i++) {
    cpu->x[i] = rand() % 16 - 8;
  }
  for (i = 0; i < nRows; i++) {
    cpu->y[i] =
    gpu->y[i] = rand() % 16 - 8;
  }
}

void delete_data(dataset_t* cpu, dataset_t* gpu) {
  free (cpu->A);
  free (cpu->x);
  free (cpu->y);
  free (gpu->y);
}

void compute_cpu(dataset_t* cpu) {
  float  alpha = cpu->alpha;
  float  beta  = cpu->beta ;
  float* A = cpu->A;
  float* x = cpu->x;
  float* y = cpu->y;
  int r, c;
  int nRows = cpu->nRows;
  int nCols = cpu->nCols;
  for (r = 0; r < nRows; r++) {
    float result = 0.0f;
    for (c = 0; c < nCols; c++) {
      result += A[nRows*c+r]*x[c];
    }
    y[r] = alpha*result + beta*y[r];
  }
}

