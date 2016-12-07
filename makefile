C = icc
CFLAGS = -Wall -O3 -g -fopenmp -lfftw3 -lm

TARGET = md_openmp

# Here we make the executable file
all: $(TARGET)

$(TARGET): $(TARGET).c
	${C} ${CFLAGS} ${TARGET}.c -o d 

# Whereas here we create the object file
#objects = ${PROG}.o
#${PROG}.o :	${PROG}.c
#	${CXX} ${CXXFLAGS} -c ${PROG}.c

# Clean
#clean:
#	rm ${objects}
