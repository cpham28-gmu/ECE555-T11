WARNINGS=-W -Wall \
    -Wcast-align -Wcast-qual -Wshadow \
    -Wwrite-strings

CFLAGS=-g -O1 $(WARNINGS)

ALL = serial

serial		:	*.cpp
				g++ $(CFLAGS) -I../kissfft/ -I./*.h *.cpp -o serial ../lib/libkissfft-float.a

clean:
	rm -f *.o 
	rm -f $(ALL)
	

