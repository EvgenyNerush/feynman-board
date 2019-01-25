lib.so: lib.cpp
	g++ -O3 -fopenmp -pthread -fPIC -shared -Wall -Wpedantic lib.cpp -o lib.so
