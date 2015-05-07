#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "common.h"

char* readFile(const char* fileName)
{
        FILE* fp;
        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error 1!\n");
                exit(1);
        }

        fseek(fp,0,SEEK_END);
        long size = ftell(fp);
        rewind(fp);

        char* buffer = (char*)malloc(sizeof(char)*(size+1));
        if(buffer  == NULL)
        {
                printf("Error 2!\n");
                fclose(fp);
                exit(1);
        }

        size_t res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error 3!\n");
                fclose(fp);
                exit(1);
        }

	buffer[size] = 0;
        fclose(fp);
        return buffer;
}

int compare(dataset_t* cpu, dataset_t* gpu) {
  int i, j;
  for (i = 0; i < cpu->nRows; ++i) {
    if (cpu->y[i] != gpu->y[i]) {
      printf ("Error: result mismatch at %d, %d\n", i, j);
      return 1;
    }
  }
  return 0;
}

double get_sec (void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


