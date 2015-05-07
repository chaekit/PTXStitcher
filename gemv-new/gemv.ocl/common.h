#ifndef COMMON_H
#define COMMON_H

#include "parboil.h"

typedef struct _tag {
  float* A;
  float* x;
  float* y;
  float  alpha;
  float  beta;
  int    nRows;
  int    nCols;
} dataset_t;

void create_data(dataset_t* cpu, dataset_t* gpu, int nRows, int nCols);
void delete_data(dataset_t* cpu, dataset_t* gpu);
void compute_cpu(dataset_t* cpu);
void compute_gpu(dataset_t* gpu, struct pb_Parameters *parameters);
int  compare(dataset_t* cpu, dataset_t* gpu);
char* readFile(const char*);
double get_sec (void);

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

#endif


