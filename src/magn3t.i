%module magn3t
%{
  #include "magn3t.cpp"
%}

%include "Eigen.i"
%include "numpy.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_complex.i>

%apply (double* IN_ARRAY1, int DIM1) {(double* seq, int n)};
%apply (double* IN_ARRAY2, int DIM1,int DIM2) {(double* seq, int n, int m)};
%apply (float* IN_ARRAY3, int DIM1, int DIM2,int DIM2) {(const float* array, int m,int n,int p)}
namespace std{
%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;
%template(ComplexVector) std::vector<std::complex<double>>;
%template(StringVector) std::vector<string>;
%template(FloatVector) std::vector<float>;
%template(VectorOfDoubleVector) std::vector<std::vector<double>>;
}



%include <std_string.i>
%include "magn3t.cpp"



//in codul sursa mereu std::complex ca sa mearga
