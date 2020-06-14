CXX = g++
LANGFLAGS = -std=c++14
#LANGFLAGS = -std=c++1y
#CXXFLAGS = -Ofast $(LANGFLAGS) -march=core-avx2 -msse
CXXFLAGS = $(LANGFLAGS) -march=core-avx2 -msse
LIBRARY=-I/opt/intel/mkl/include

all: finedex_benchmark

finedex_benchmark:
	$(CXX) $(CXXFLAGS) -o ./build/finedex_benchmark ./finedex_benchmark.cpp $(LIBRARY) -pthread
