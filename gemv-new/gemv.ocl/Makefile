# (c) 2010 The Board of Trustees of the University of Illinois.

OPENCL_INCLUDE_PATH=/opt/AMDAPP/include

CFLAGS=-c -g -O3 -I$(OPENCL_INCLUDE_PATH) -I.

OBJS=main.o cpu.o support.o gpu.o args.o parboil_opencl.o
APP=sgemv
BINS=$(APP).ocl

all: $(BINS)

$(APP).ocl: $(OBJS)
	gcc -o $@ $^ -lOpenCL

%.o: %.c
	gcc -o $@ $< $(CFLAGS) -c

clean:
	rm -rf $(OBJS) $(BINS)

