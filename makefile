CXX = g++

all : calcEM.exe calcMassFit.exe

jk.o: jk.cpp ; $(CXX) -c -I. jk.cpp

calcEM.o: calcEM.cpp ; $(CXX) -c -I. calcEM.cpp

calcEM.exe: calcEM.o jk.o ; $(CXX) -o calcEM.exe \
	calcEM.o jk.o\

calcMassFit.o: calcMassFit.cpp ; $(CXX) -c -I. calcMassFit.cpp

calcMassFit.exe: calcMassFit.o ; $(CXX) -o calcMassFit.exe \
	calcMassFit.o\

clean: 
	rm -f *.o
	rm -f *.exe
	rm -f *~

