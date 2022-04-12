CPP=/usr/bin/g++ 

CPPFLAGS=-O2 -std=c++11
export CPPFLAGS

name=magn3t

PYTHON_INCLUDE = -I/usr/include/python3.7m/ 
EIGEN_INCLUDE = -I /usr/local/include/eigen3 
LIBS=#

#$(CPP) $(name).cpp -o $(name).o $(CPPFLAGS) $(PYTHON_INCLUDE) $(EIGEN_INCLUDE)

all:
	swig3.0 -c++ -python $(name).i
	g++ $(CPPFLAGS) $(LIBS) -fPIC $(PYTHON_INCLUDE) $(EIGEN_INCLUDE) -c $(name)_wrap.cxx -o $(name)_wrap.o 
	g++ $(PYTHONL) $(LIBS) -pthread -fopenmp -shared $(name)_wrap.o -o _magn3t.so  
	rm $(name)_wrap.cxx $(name)_wrap.o

link:
	rm -f $(name).o && \
	$(CPP) -o $(name).o *.o $(LIBS)
run:  
	./$(name).o



	
clean:
	rm -f $(name).o

