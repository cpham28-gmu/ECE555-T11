WARNINGS=-W -Wall -Waggregate-return \
    -Wcast-align -Wcast-qual -Wshadow \
    -Wwrite-strings

CFLAGS=-g -O1 $(WARNINGS)

ALL = test

test		:	*.cpp
				g++ $(CFLAGS) -fopenmp -I../kissfft/ -I./*.h *.cpp -o test -lkissfft-float-openmp

clean:
	rm -f *.o 
	rm -f $(ALL)
	

