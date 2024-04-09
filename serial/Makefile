WARNINGS=-W -Wall -Wwrite-strings\
    -Wcast-align -Wcast-qual -Wshadow
    

CFLAGS=-g -O1 $(WARNINGS)

ALL = serial

serial 		:	*.cpp
				g++ $(CFLAGS) -I../kissfft/ -I./*.h *.cpp -o serial -lkissfft-float

clean:
	rm -f *.o 
	rm -f $(ALL)
	

