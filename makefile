CFLAGS = -Wall -O3 -g -fopenmp 
LIBS = -lfftw3 -lm

TARGET = md_openmp

all: d

$(TARGET).o: $(TARGET).c

d: $(TARGET).o
	$(CC) $(CFLAGS) $(LIBS) $(TARGET).o -o $@

# Clean
clean:
	rm -f *.o

# Here we make the executable file
#all: d
#
#md_openmp.o: md_openmp.c
#	
#d: md_openmp.o
#	${CC} $(CFLAGS) md_openmp.o -o $@  $(LIBS) 
#
# Whereas here we create the ob4ject file
#objects = ${PROG}.o
#${PROG}.o :	${PROG}.c
#	${CXX} ${CXXFLAGS} -c ${PROG}.c

