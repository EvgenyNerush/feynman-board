lib.so: lib.cpp
	g++ -fPIC -shared -Wall -Wpedantic lib.cpp -o lib.so
