# (c) 2010 The Board of Trustees of the University of Illinois.

CUDA_PATH=/usr/local/cuda

CFLAGS=-c -g -O3 -I$(CUDA_PATH)/include -I.
LDFLAGS=-lcuda -L/usr/lib64/nvidia

OBJS=main.o cpu.o support.o gpu.o
APP=sgemv
BINS=$(APP).cuda

all: $(BINS)

$(APP).cuda: $(OBJS)
	nvcc -o $@ $^ $(LDFLAGS)

%.o: %.c
	gcc -o $@ $< $(CFLAGS) -c

%.o: %.cc
	nvcc -o $@ $< $(CFLAGS) -c

clean:
	rm -rf $(OBJS) $(BINS)

