CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp
PROG = dnnmnist

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

.PHONY: all clean

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)
