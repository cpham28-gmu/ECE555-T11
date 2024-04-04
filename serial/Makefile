WARNINGS=-W -Wall -Waggregate-return \
    -Wcast-align -Wcast-qual -Wshadow \
    -Wwrite-strings

CFLAGS=-g -O1 $(WARNINGS)

ALL = test

test		:	test.cpp
				g++ $(CFLAGS) -fopenmp -I../kissfft/ test.cpp -o test  -L../lib/  -lkissfft-float-openmp

clean:
	rm -f *.o 
	rm -f $(ALL)
	

