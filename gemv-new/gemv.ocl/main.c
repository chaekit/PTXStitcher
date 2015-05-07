/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"

int main(int argc, char *argv[]) {
  struct pb_Parameters *parameters;
  int nRows, nCols;
  int retval;

  /* Read input parameters */
  parameters = pb_ReadParameters(&argc, argv);
  if (parameters == NULL) {
    fprintf(stderr, "Error in parameters.\n");
    exit(1);
  }

  if (argc == 2) {
    nRows = atoi(argv[1]);
    nCols = atoi(argv[1]);
  } else if (argc == 3) {
    nRows = atoi(argv[1]);
    nCols = atoi(argv[2]);
  } else {
    nRows = 4096;
    nCols = 4096;
  }

  dataset_t cpu, gpu;
  create_data(&cpu, &gpu, nRows, nCols);
  compute_cpu(&cpu);
  compute_gpu(&gpu, parameters);
  retval = compare(&cpu, &gpu);
  delete_data(&cpu, &gpu);
  pb_FreeParameters(parameters);
  return retval;
}

