CPP=/usr/bin/g++ 

CPPFLAGS=-O2 -std=c++11
export CPPFLAGS

nume=tomo3

PYTHON_INCLUDE = -I/usr/include/python3.7m/ 
EIGEN_INCLUDE = -I /usr/local/include/eigen3 
LIBS=#
all: 
	$(CPP) $(nume).cpp -o $(nume).o $(CPPFLAGS) $(PYTHON_INCLUDE) $(EIGEN_INCLUDE)
	
link:
	rm -f $(nume).o && \
	$(CPP) -o $(nume).o *.o $(LIBS)
run:  
	./$(nume).o


interface:
	swig3.0 -c++ -python $(nume).i
	g++ $(CPPFLAGS) $(LIBS) -fPIC $(PYTHON_INCLUDE) $(EIGEN_INCLUDE) -c $(nume)_wrap.cxx -o $(nume)_wrap.o 
	g++ $(PYTHONL) $(LIBS) -pthread -fopenmp -shared $(nume)_wrap.o -o _tomo.so  
	rm $(nume)_wrap.cxx $(nume)_wrap.o

	
clean:
	rm -f $(nume).o

