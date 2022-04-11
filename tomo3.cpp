#include <unistd.h>
#include <string>
#include <iostream>
#include <iomanip>  
#include <math.h>
#include <fstream>
#include <cmath>
#include <complex>
#include <vector>
#include <deque>
#include <ctime>
#include <time.h>
#include <queue>
#include <chrono>
#include <random>

#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Core"




using namespace std;

typedef Eigen::Vector3d Vec3;
typedef Eigen::VectorXd VecX;
typedef Eigen::Vector3i Vec3i;

double random_double(double a,double b){
    return a+(b-a)*(double)rand()/(double)RAND_MAX;
    
}	
double pi=3.14159265358979;

vector<float> MRCtoVector(string fileName);

class cubee{
	public:
	int Nx;
	int Ny;
	int Nz;
	int N_tot;
	double xx;
	double Volume;
	double Surface;
	

	vector<float> Val;
	int modulo(int i,int nn){
		return (i%nn+nn)%nn;
	}
	int modulo(int i,int nn)const{
		return (i%nn+nn)%nn;
	}
	int index(int i, int j,int k){//column major
		return modulo(i,Nx)+Nx*(modulo(j,Ny)+Ny*modulo(k,Nz));
	}
	
	int vecini(int centru,int vecX,int vecY,int vecZ){
		return modulo((centru+vecX+vecY*Nx+vecZ*Ny*Nx),N_tot); 

	}	

	Vec3i index3D(int ind){
		Vec3i  rez;
		//~ int k=round(float(ind)/(Nx*Ny));
		//~ int j=round((float)(ind-k*Nx*Ny)/Nx);
		//~ int i=ind-Nx*(j+Ny*k);
		int i=ind%Nx;
		int j=(int)(float(ind)/Nx)%(Ny);
		int k=(int)1.*ind/(Nx*Ny);
		
		rez<<i,j,k;
		//rez.push_back(j);rez.push_back(k);
		return rez;
	}
	
	//~ int index(int i, int j,int k){//column major
		//~ return i+Nx*j+Nx*Ny*k;
	//~ }
	
	//~ int index(int i, int j,int k)const {//column major
		//~ return modulo(i,Nx)+Nx*(modulo(j,Ny)+Ny*modulo(k,Nz));
	//~ }
	
	void test(){
		double dif=0;
		for(int i1=0;i1<Nx;i1++)
		for(int i2=0;i2<Ny;i2++){
		for(int i3=0;i3<Nz;i3++){
			auto x=index(i1,i2,i3);
			auto v=index3D(x);
			dif+=(v[0]-i1)+(v[1]-i2)+v[2]-i3;
			
		}}
		
		//cout<<dif<<endl;
	}
	int index2(int i, int j,int k){//column major
		return i*Ny*Nz+j*Nz+k;// modulo(k,Nz)+Nz*(modulo(j,Ny)+Ny*modulo(i,Nx));
	}

	cubee(int Nx,int Ny,int Nz){
		this->Nx=Nx;
		this->Ny=Ny;
		this->Nz=Nz;
		N_tot=Nx*Ny*Nz;
		Val.resize(N_tot);
		
	}
	
	cubee(vector<float> v,int Nx,int Ny,int Nz){
		this->Nx=Nx;
		this->Ny=Ny;
		this->Nz=Nz;
		N_tot=Nx*Ny*Nz;
		Val=v; 
	}
	
	cubee(string fileName){
		fstream fin(fileName.c_str(), std::ios::in | std::ios::binary);
		int32_t dim[3];
		fin.read(reinterpret_cast<char *>(dim), 3*sizeof(int32_t));
		//cout<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<endl;
		
		this->N_tot = dim[0]*dim[1]*dim[2];
		//cubee tomo(dim[0],dim[1],dim[2]);
		this->Nx=dim[0];
		this->Ny=dim[1];
		this->Nz=dim[2];
		Val.resize(N_tot);
		int32_t mode;
		fin.read(reinterpret_cast<char*>(&mode), sizeof(mode));
		int32_t nxstart;
		fin.read(reinterpret_cast<char*>(&nxstart), sizeof(int32_t));
		//cout<<nxstart<<endl;
		int32_t nystart;
		fin.read(reinterpret_cast<char*>(&nystart), sizeof(int32_t));
		//cout<<nxstart<<endl;
		int32_t nzstart;
		fin.read(reinterpret_cast<char*>(&nzstart), sizeof(int32_t));
		//cout<<nzstart<<endl;
		int32_t mx;
		fin.read(reinterpret_cast<char*>(&mx), sizeof(int32_t));
		//cout<<mx<<endl;
		int32_t my;
		fin.read(reinterpret_cast<char*>(&my), sizeof(int32_t));
		//cout<<my<<endl;
		int32_t mz;
		fin.read(reinterpret_cast<char*>(&mz), sizeof(int32_t));
		//cout<<mz<<endl;
		int32_t dim_cell[3];
		fin.read(reinterpret_cast<char *>(dim_cell), 3*sizeof(int32_t));
		//cout<<dim_cell[0]<<" "<<dim_cell[1]<<" "<<dim_cell[2]<<endl;
		fin.seekg(1024);
		
		int buf_len = sizeof(float) * N_tot;
		
		float *fvol = new float[buf_len];
		fin.read(reinterpret_cast<char*>(fvol), buf_len);
		fin.close();
		
		for(int i=0;i<this->N_tot;i++){
			this->Val[i]=fvol[i];
		}
		delete fvol;
	}
	
	//~ ~cubee(){
	//~ Val.clear();
	//~ }
	
	void close(){
	delete this;
	}
	vector<float> getVal(){
		return Val;
	}
	
	//~ void print2(vector<float> a1 ){
		//~ cout<<a1.size()<<endl;
	//~ }


	float get(int i1,int i2, int i3){
		return(Val[index(i1,i2,i3)]);
	}
	
	float get(int i,int i1,int i2, int i3){
		return(Val[vecini(i,i1,i2,i3)]);
	}
	
	float getSafe(int i1,int i2,int i3){
		if(i1>0 && i1<Nx && i2>0 && i2<Ny && i3>0 && i3<Nz){
			return get(i1,i2,i3);
		}else{
			return 0;
		}
	}
	
	float get(int ind){
		return Val[ind];
	}
	void set(int ind,double val){
		Val[ind]=val;
	}
	
	void set(int i1,int i2, int i3,double v){
		Val[index(i1,i2,i3)]=v;
	}
	
	void setSafe(int i1,int i2,int i3,double val){
		if(i1>0 && i1<Nx && i2>0 && i2<Ny && i3>0 && i3<Nz){
			set(i1,i2,i3,val);
		}
	}
	
	void normalize(){
		float valMax=-1;
		for(int i=0;i<N_tot;i++){
		if(Val[i]>=valMax){
			valMax=Val[i];
		}
	}
	for(int i=0;i<N_tot;i++){
		Val[i]/=valMax;
	}
	
	
	}


	float operator()(int i1,int i2,int i3)  {
		return Val[index(i1,i2,i3)];
	}
	

	cubee operator - (cubee other){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]-=other.get(i);
		}
		return out;
	}
	cubee operator - (float x){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]-=x;
			if(out.Val[i]<0){
				out.Val[i]=0;
			}
		}
		return out;
	}
	
	cubee operator + (cubee other){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]+=other.get(i);
		}
		return out;
	}
	cubee operator + (float x){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]+=x;
			if(out.Val[i]<0){
				out.Val[i]=0;
			}
		}
		return out;
	}	
	cubee operator * (float x){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]*=x;
			if(out.Val[i]<0){
				out.Val[i]=0;
			}
		}
		return out;
	}

	cubee operator / (float x){
		cubee out=(*this);
		for(int i=0;i<out.N_tot;i++){
			out.Val[i]/=x;
			if(out.Val[i]<0){
				out.Val[i]=0;
			}
		}
		return out;
	}
	
	cubee copy(){
		cubee cc(Nx,Ny,Nz);
		cc.Val=Val;
		return cc;
	}
	
	
	void erode(int culoare,int n){
	for(int q=0;q<n;q++){
		
		vector<int> puncteSuprafata;
		
		for(int i=0;i<N_tot;i++){
			if(Val[i]==culoare){
				if(
				get(vecini(i,1,0,0))!=culoare||
				get(vecini(i,-1,0,0))!=culoare||
				get(vecini(i,0,1,0))!=culoare||
				get(vecini(i,0,-1,0))!=culoare||
				get(vecini(i,0,0,1))!=culoare||
				get(vecini(i,0,0,-1))!=culoare){
					puncteSuprafata.push_back(i);
				}
			
			}
			
			
		
		}
		
		
		
		for(int i=0;i<puncteSuprafata.size();i++){
			Val[puncteSuprafata[i]]=0;
		}
	
	
	
	
	}

	}
	
	void addSphere(float x,float y,float z, float raza,float smo=0){
		if(smo==0){
			for(int i=0;i<Nx;i++)
			for(int j=0;j<Ny;j++)
			for(int k=0;k<Nz;k++){
			double r=sqrt(pow(i-x,2)+pow(j-y,2)+pow(k-z,2));
			if(r<=raza){
				set(i,j,k,1);
			}	
				
				
			}
		}else{
			for(int i=0;i<Nx;i++)
			for(int j=0;j<Ny;j++)
			for(int k=0;k<Nz;k++){
			double r=sqrt(pow(i-x,2)+pow(j-y,2)+pow(k-z,2));
			
			double v=1./(1+exp((r-raza)/smo));
			
			double val_init=get(i,j,k);
			if(val_init+v<=1){
				set(i,j,k,val_init+v);
			}else{
				set(i,j,k,1);
			}
		
		
			}
		}
		
		
	}
	
	double interpolation(double x,double y,double z){
		int i1=(int)(x-0);
		int i2=(int)(y-0);
		int i3=(int)(z-0);
		i1=modulo(i1,Nx);
		i2=modulo(i2,Ny);
		i3=modulo(i3,Nz);
		
		//~ cout<<x<<" "<<y<<" "<<z<<endl;
		double xd=(x-i1);
		double yd=(y-i2);
		double zd=(z-i3);
		
		double c00=get(i1,i2,i3)*(1-xd)+get(i1+1,i2,i3)*xd;
		double c01=get(i1,i2,i3+1)*(1-xd)+get(i1+1,i2,i3+1)*xd;
		double c10=get(i1,i2+1,i3)*(1-xd)+get(i1+1,i2+1,i3)*xd;
		double c11=get(i1,i2+1,i3+1)*(1-xd)+get(i1+1,i2+1,i3+1)*xd;
		
		double c0=c00*(1-yd)+c10*yd;
		double c1=c01*(1-yd)+c11*yd;
		
		return c0*(1-zd)+c1*zd;
	}
	
	cubee fillInterpolation(int nx,int ny,int nz){
		cubee c(nx,ny,nz);
		
		double dx=1.*(Nx-1)/(nx-1);
		double dy=1.*(Ny-1)/(ny-1);
		double dz=1.*(Nz-1)/(nz-1);
		double x,y,z;
		for(int i=0;i<nx;i++)
		for(int j=0;j<ny;j++)
		for(int k=0;k<nz;k++){
			x=i*dx;
			y=j*dy;
			z=k*dz;
			//~ cout<<interpolation(x,y,z)<<endl;
			c.set(i,j,k,interpolation(x,y,z));
            //~ int i1=(int)(x-0);
            //~ int i2=(int)(y-0);
            //~ int i3=(int)(z-0);
            //~ i1=modulo(i1,Nx);
            //~ i2=modulo(i2,Ny);
            //~ i3=modulo(i3,Nz);

            //~ double xd=(x-i1);
            //~ double yd=(y-i2);
            //~ double zd=(z-i3);
            
            //~ double c00=get(i1,i2,i3)*(1-xd)+get(i1+1,i2,i3)*xd;
            //~ double c01=get(i1,i2,i3+1)*(1-xd)+get(i1+1,i2,i3+1)*xd;
            //~ double c10=get(i1,i2+1,i3)*(1-xd)+get(i1+1,i2+1,i3)*xd;
            //~ double c11=get(i1,i2+1,i3+1)*(1-xd)+get(i1+1,i2+1,i3+1)*xd;
            
            //~ double c0=c00*(1-yd)+c10*yd;
            //~ double c1=c01*(1-yd)+c11*yd;
            
            //~ return c0*(1-zd)+c1*zd;
         }
        return c;
        
	}
	
	cubee fillInterpolation(cubee c){
		
		
		double dx=(c.Nx-1)/(Nx-1);
		double dy=(c.Ny-1)/(Ny-1);
		double dz=(c.Nz-1)/(Nz-1);
		double x,y,z;
		for(int i=0;i<Nx;i++)
		for(int j=0;j<Ny;j++)
		for(int k=0;k<Ny;k++){
			x=i*dx;
			y=j*dy;
			z=k*dz;
			
			set(i,j,k,c.interpolation(x,y,z));
            //~ int i1=(int)(x-0);
            //~ int i2=(int)(y-0);
            //~ int i3=(int)(z-0);
            //~ i1=modulo(i1,Nx);
            //~ i2=modulo(i2,Ny);
            //~ i3=modulo(i3,Nz);

            //~ double xd=(x-i1);
            //~ double yd=(y-i2);
            //~ double zd=(z-i3);
            
            //~ double c00=get(i1,i2,i3)*(1-xd)+get(i1+1,i2,i3)*xd;
            //~ double c01=get(i1,i2,i3+1)*(1-xd)+get(i1+1,i2,i3+1)*xd;
            //~ double c10=get(i1,i2+1,i3)*(1-xd)+get(i1+1,i2+1,i3)*xd;
            //~ double c11=get(i1,i2+1,i3+1)*(1-xd)+get(i1+1,i2+1,i3+1)*xd;
            
            //~ double c0=c00*(1-yd)+c10*yd;
            //~ double c1=c01*(1-yd)+c11*yd;
            
            //~ return c0*(1-zd)+c1*zd;
         }
        return c;
        
	}
	
	
};

cubee readMRC(string);
void writeMRC(cubee,string);
void applyThreshold(cubee &c,double th){
	
	//#pragma omp parallel for num_threads(1) collapse(1) shared(c,th) 
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++){
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)>th){
			c.set(i1,i2,i3,10);
		}else{
			c.set(i1,i2,i3,0);
		}
	}}
	
	//~ #pragma omp parallel for num_threads(1)
	//~ for(int i=0;i<c.Val.size();i++){
		//~ if(c.get(i)>th){
			//~ c.set(i,1);
		//~ }else{
			//~ c.set(i,0);
		//~ }
	//~ }
	
	
}
float fd(double x,double r,double t){
	return (float)1./(1+exp((x-r)/t));
}
double volume(cubee& c, int culoare){
	int vol=0;
	for(int i=0;i<c.N_tot;i++){
		if(c.Val[i]==culoare){
			vol++;
		}
	}

	return vol;
}

double volume(cubee& c){
	int vol=0;
	for(int i=0;i<c.N_tot;i++){
		vol+=c.get(i);
	}

	return vol;
}

double surface(cubee& c, int culoare){
	double surf=0;
	for(int i=0;i<c.N_tot;i++){
			if(c.Val[i]==culoare){
				if(
				c.get(c.vecini(i,1,0,0))!=culoare||
				c.get(c.vecini(i,-1,0,0))!=culoare||
				c.get(c.vecini(i,0,1,0))!=culoare||
				c.get(c.vecini(i,0,-1,0))!=culoare||
				c.get(c.vecini(i,0,0,1))!=culoare||
				c.get(c.vecini(i,0,0,-1))!=culoare){
					surf++;
				}
		}
	}
	
	return surf;
}

double surfaceModificat(cubee& c, int culoare){
	double surf=0;
	double w1=0.894,w2=1.3409,w3=1.5879,w4=2.,w5=8./3,w6=10./3,w7=1.79,w8=2.68,w9=4.08;
	vector<int> distCat;
	for(int i=0;i<9;i++){
		distCat.push_back(0);
	}
	int contorPuncte=0;
	for(int i=0;i<c.N_tot;i++){
			Vec3 vi({0,0,0});
			int contorVec=0;
			if(c.Val[i]==culoare){
				
				if(c.get(c.vecini(i,1,0,0))!=culoare){
					vi[0]+=1;
					contorVec+=1;
				}
				if(c.get(c.vecini(i,-1,0,0))!=culoare){
					vi[0]+=-1;
					contorVec+=1;
				}
				if(c.get(c.vecini(i,0,1,0))!=culoare){
					vi[1]+=1;
					contorVec+=1;
				}
				if(c.get(c.vecini(i,0,-1,0))!=culoare){
					vi[1]+=-1;
					contorVec+=1;
				}
				if(c.get(c.vecini(i,0,0,1))!=culoare){
					vi[2]+=1;
					contorVec+=1;
				}
				if(c.get(c.vecini(i,0,0,-1))!=culoare){
					vi[2]+=-1;
					contorVec+=1;
				}
			int cat=0;
			if(contorVec==1){
				surf+=w1;
				cat=1;
			}
			if(contorVec==2){
				if(vi.norm()<0.1){
					cat=7;
					surf+=w7;
				}else{
					cat=2;
					surf+=w2;
					
				}
			}
			if(contorVec==3){
				if(vi.norm()<1.01){
					cat=4;
					surf+=w4;
				}else{
					cat=3;
					surf+=w3;
					
				}
			}
			
			if(contorVec==4){
				if(vi.norm()<0.1){
					cat=8;
					surf+=w8;
				}else{
					cat=5;
					surf+=w5;
					
				}
			}
			
			if(contorVec==5){
				cat=6;
				surf+=w6;
			}
			
			if(contorVec==6){
				cat=9;
				surf+=w9;
			}
			
			if(cat!=0){
				contorPuncte+=1;
				//~ cout<<cat<<": "<<vi.norm()<<" - "<<vi.transpose()<<endl;
				distCat[cat-1]+=1;
			}
			
		}
	}
	
	//~ ofstream fis("distCat");
	for(int i=0;i<9;i++){
	cout<<" "<<1.*distCat[i]/contorPuncte;
	}
	
	return surf;
}

void applyThreshold_linear(cubee &c, double th){

		for(int i=0;i<c.N_tot;i++){
			if(c.get(i)>th){
				c.set(i,10);
			}else{
				c.set(i,0);
			}
		}	
	}
void applyThreshold_linear(cubee &c, double th, float culoare){
		//#pragma omp parallel
		{
		//#pragma omp for simd
		for(int i=0;i<c.N_tot;i++){
			if(c.get(i)>th){
				c.set(i,culoare);
			}else{
				c.set(i,0);
			}
		}
		
		}	
	}
	
Vec3i findValue(cubee &c, int value){
	for(int i=0;i<c.N_tot;i++){
		if(c.Val[i]==value){
			//cout<<i<<endl;
				return c.index3D(i);
		}
	}
	
	return Vec3i(-1,-1,-1);
	
}



double fill(cubee &c,int i1,int i2,int i3,int target,int newcolor){
	
	int contor=0;
	if(c.get(i1,i2,i3)!=target){
	}else{
		c.set(i1,i2,i3,newcolor);
		deque<Vec3> vc;
		vc.push_back(Vec3(i1,i2,i3));
		
		while(vc.size()!=0){
			Vec3 x=vc.front();
			vc.pop_front();
			contor++;
				
			if(x(0)<c.Nx-1 && x(0)>=1 && x(1)<c.Ny-1 && x(1)>=1 && x(2)<c.Nz-1 && x(2)>=1){
			if(c.get(x(0)+1,x(1),x(2))==target){
				c.set(x(0)+1,x(1),x(2),newcolor);
				vc.push_back(Vec3(x(0)+1,x(1),x(2)));
			}
			if( c.get(x(0)-1,x(1),x(2))==target){
				c.set(x(0)-1,x(1),x(2),newcolor);
				vc.push_back(Vec3(x(0)-1,x(1),x(2)));
			}
			if(c.get(x(0),x(1)+1,x(2))==target){
				c.set(x(0),x(1)+1,x(2),newcolor);
				vc.push_back(Vec3(x(0),x(1)+1,x(2)));
			}
			
			if(c.get(x(0),x(1)-1,x(2))==target){
				c.set(x(0),x(1)-1,x(2),newcolor);
				vc.push_back(Vec3(x(0),x(1)-1,x(2)));
			}
			if(c.get(x(0),x(1),x(2)+1)==target){
				c.set(x(0),x(1),x(2)+1,newcolor);
				vc.push_back(Vec3(x(0),x(1),x(2)+1));
			}
			
			if(c.get(x(0),x(1),x(2)-1)==target){
				c.set(x(0),x(1),x(2)-1,newcolor);
				vc.push_back(Vec3(x(0),x(1),x(2)-1));
			}
		
		}
		
		}
	}
	return contor;
	
	
}

double fill_particule(cubee& c,int startColor){
	int culoare=11;
	
	ofstream vol("volumeTest");

	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)==startColor){
			fill(c,i1,i2,i3,startColor,culoare);
			vol<<culoare<<" "<<volume(c,culoare)<<endl;
			culoare++;
		}
	}
		
		return  culoare-11;
	
}

vector<double> fill_particule_volume(cubee& c,int startColor){
	int culoare=11;
	vector<double> volume1;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)==startColor){
			volume1.push_back(fill(c,i1,i2,i3,startColor,culoare));
			culoare++;
		}
	}
		
		return volume1;
	
}

vector<int> volumeList(cubee& c){
	
	vector<int> v(32);
	double culoareMax=-1;
	
	for(int i=0;i<c.N_tot;i++){
		
		int culoare=c.get(i);
		if(culoare>=culoareMax){
			culoareMax=culoare;
		}
		
		if(culoare>=v.size()){
			v.resize(culoare+1);
		}
		
		v[culoare]+=1;
		
		
	
	}
	//~ v.resize(culoareMax+1);
	return v;
	
	
}

vector<double> fill_particule_volume_2(cubee& c,int startColor){
	int culoare=11;
	vector<double> volume1;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)==startColor){
			volume1.push_back(fill(c,i1,i2,i3,startColor,-1));
			startColor++;
		}
		
	}
		
		return volume1;
	
}


double fillParticlesRandom(cubee& c,int startColor){
	cubee init=c;
	int culoare=11;
	vector<int>culori;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(init.get(i1,i2,i3)==startColor){
			fill(init,i1,i2,i3,startColor,culoare);
			culori.push_back(culoare);
			culoare++;
			
		}
	}
	
	//~ ofstream testRandom("testRandom");
	//~ for(int i=0;i<culori.size();i++){
	//~ testRandom<<i<<" "<<culori[i]<<endl;
	//~ }testRandom<<endl;
	srand(time(NULL));
	random_shuffle(culori.begin(),culori.end());
	//~ for(int i=0;i<culori.size();i++){
	//~ testRandom<<i<<" "<<culori[i]<<endl;
	//~ }
	int i=0;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)==startColor){
			fill(c,i1,i2,i3,startColor,culori[i]);
			i++;
			
		}
	}
	
	
		
	return  culoare-11;
	
}

double fillParticlesRandom_2(cubee& c,int startColor,int startColor2){
	cubee init=c;
	int culoare=startColor2;
	vector<int>culori;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(init.get(i1,i2,i3)==startColor){
			fill(init,i1,i2,i3,startColor,culoare);
			culori.push_back(culoare);
			culoare++;
			
		}
	}
	
	//~ ofstream testRandom("testRandom");
	//~ for(int i=0;i<culori.size();i++){
	//~ testRandom<<i<<" "<<culori[i]<<endl;
	//~ }testRandom<<endl;
	srand(time(NULL));
	random_shuffle(culori.begin(),culori.end());
	//~ for(int i=0;i<culori.size();i++){
	//~ testRandom<<i<<" "<<culori[i]<<endl;
	//~ }
	int i=0;
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++){
		if(c.get(i1,i2,i3)==startColor){
			fill(c,i1,i2,i3,startColor,culori[i]);
			i++;
			
		}
	}
	
	
		
	return  culoare-startColor2;
	
}
void erode(cubee &c,int culoare,int n){
	for(int q=0;q<n;q++){
		
		vector<int> puncteSuprafata;
		
		for(int i=0;i<c.N_tot;i++){
			if(c.Val[i]==culoare){
				if(
				c.get(c.vecini(i,1,0,0))!=culoare||
				c.get(c.vecini(i,-1,0,0))!=culoare||
				c.get(c.vecini(i,0,1,0))!=culoare||
				c.get(c.vecini(i,0,-1,0))!=culoare||
				c.get(c.vecini(i,0,0,1))!=culoare||
				c.get(c.vecini(i,0,0,-1))!=culoare){
					puncteSuprafata.push_back(i);
				}
			
			}
			
			Vec3i ind=c.index3D(i);
			
			//~ if(ind[0]==0||ind[0]==c.Nx-1||ind[1]==0||ind[1]==c.Ny-1||ind[2]==0||ind[2]==c.Nz-1){
				//~ puncteSuprafata.push_back(i);
			//~ }
			
			
		
		}
		
		
		
		for(int i=0;i<puncteSuprafata.size();i++){
			c.Val[puncteSuprafata[i]]=0;
		}
	
		string nume="sfereErode"+to_string(q)+".mrc";
		//writeMRC(c,nume);
		//cubee d=c;
		//double c1,c2;
		//c1=volume(d,1);
		//c2=fill_particule(d,1);
		//cout<<"erode "<<q<<" "<<c1<<" "<<c2<<endl;


}

}


void dilate(cubee &c,int culoare,int n){
	for(int q=0;q<n;q++){		
		vector<int> puncteSuprafata;
		
		for(int i=0;i<c.N_tot;i++){
			if(c.Val[i]!=culoare){
				if(
				c.get(c.vecini(i,1,0,0))==culoare||
				c.get(c.vecini(i,-1,0,0))==culoare||
				c.get(c.vecini(i,0,1,0))==culoare||
				c.get(c.vecini(i,0,-1,0))==culoare||
				c.get(c.vecini(i,0,0,1))==culoare||
				c.get(c.vecini(i,0,0,-1))==culoare){
					puncteSuprafata.push_back(i);
				}			
			}
			
			Vec3i ind=c.index3D(i);
			
			//~ if(ind[0]==0||ind[0]==c.Nx-1||ind[1]==0||ind[1]==c.Ny-1||ind[2]==0||ind[2]==c.Nz-1){
				//~ puncteSuprafata.push_back(i);
			//~ }			
		
		}		
		for(int i=0;i<puncteSuprafata.size();i++){
			c.Val[puncteSuprafata[i]]=culoare;
		}
	
		string nume="sfereDilate"+to_string(q)+".mrc";
		//writeMRC(c,nume);
		cubee d=c;
		double c1,c2;
		c1=volume(d,1);
		c2=fill_particule(d,1);
		//cout<<"erode "<<q<<" "<<c1<<" "<<c2<<endl;	
	}
}

void dilate2(cubee &c,int culoare,int n){
	for(int q=0;q<n;q++){		
		vector<int> puncteSuprafata;
		cubee old=c;
		for(int i=0;i<c.N_tot;i++){
			if(old.Val[i]!=culoare){
				if(
				old.get(c.vecini(i,1,0,0))==culoare||
				old.get(c.vecini(i,-1,0,0))==culoare||
				old.get(c.vecini(i,0,1,0))==culoare||
				old.get(c.vecini(i,0,-1,0))==culoare||
				old.get(c.vecini(i,0,0,1))==culoare||
				old.get(c.vecini(i,0,0,-1))==culoare){
					//puncteSuprafata.push_back(i);
					c.set(i,culoare);
				}			
			}
			
		
		}		
	}
	//cout<<"dilate"<<endl;
}

void erode2(cubee &c,int culoare,int n){
	
	for(int q=0;q<n;q++){		
		vector<int> puncteSuprafata;
		cubee old=c;
		for(int i=0;i<c.N_tot;i++){
			if(old.Val[i]==culoare){
				if(
				old.get(c.vecini(i,1,0,0))!=culoare||
				old.get(c.vecini(i,-1,0,0))!=culoare||
				old.get(c.vecini(i,0,1,0))!=culoare||
				old.get(c.vecini(i,0,-1,0))!=culoare||
				old.get(c.vecini(i,0,0,1))!=culoare||
				old.get(c.vecini(i,0,0,-1))!=culoare){
					c.set(i,0);
				}
			
			}
			
			
		
		}	
	}
	//cout<<"erode"<<endl;
}


void print2(vector<float> a1,int Nx,int Ny,int Nz ){
		cubee c(a1,Nx,Ny,Nz);
	}
vector<float> unVector(int n){
	vector<float> v(n);
	for(int i=0;i<v.size();i++){
		v[i]=i;
	}
	return v;
}

void erodeGrayscale(cubee &c,int n){
	
	for(int q=0;q<n;q++){
		cubee d=c;
			
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double min=10000;
			double val;
			
			//~ for(int i1=-1;i1<=1;i1++)
			//~ for(int i2=-1;i2<=1;i2++)
			//~ for(int i3=-1;i3<=1;i3++)
			//~ {val=d.get(d.vecini(i,i1,i2,i3));
			//~ av+=val;
				//~ if(val<min)min=val;
			//~ }
			
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,k,0,0));
				if(val<min)min=val;
			}
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,0,k,0));
				if(val<min)min=val;
			}
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,0,0,k));
				if(val<min)min=val;
			}
			c.set(i,min);
		}
	}
}

void erodeGrayscaleCheb(cubee &c,int n){
	for(int q=0;q<n;q++){
		cubee d=c;
			
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double min=10000;
			double val;
			
			//~ for(int i1=-1;i1<=1;i1++)
			//~ for(int i2=-1;i2<=1;i2++)
			//~ for(int i3=-1;i3<=1;i3++)
			//~ {val=d.get(d.vecini(i,i1,i2,i3));
			//~ av+=val;
				//~ if(val<min)min=val;
			//~ }
			
			for(int i1=-1;i1<=1;i1++)
			for(int i2=-1;i2<=1;i2++)
			for(int i3=-1;i3<=1;i3++){
				val=d.get(d.vecini(i,i1,i2,i3));
				if(val<min)min=val;
			}
			
			c.set(i,min);
		

}
}
}

void dilateGrayscale(cubee &c,int n){
	
	for(int q=0;q<n;q++){
		cubee d=c;
			
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double max=-10000;
			double val;
			
			//~ for(int i1=-1;i1<=1;i1++)
			//~ for(int i2=-1;i2<=1;i2++)
			//~ for(int i3=-1;i3<=1;i3++)
			//~ {val=d.get(d.vecini(i,i1,i2,i3));
			//~ av+=val;
				//~ if(val>max)max=val;
			//~ }
			
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,k,0,0));
				if(val>max)max=val;
			}
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,0,k,0));
				if(val>max)max=val;
			}
			for(int k=-1;k<=1;k++){
				val=d.get(d.vecini(i,0,0,k));
				if(val>max)max=val;
			}
			c.set(i,max);
		

}
}
}

void dilateGrayscaleCheb(cubee &c,int n){
	for(int q=0;q<n;q++){
		cubee d=c;
		vector<double> ad={0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5,0.5,1,0.5,1,1,1,0.5,1,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,0.5};
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double max=-10000;
			double val;
			
			//~ for(int i1=-1;i1<=1;i1++)
			//~ for(int i2=-1;i2<=1;i2++)
			//~ for(int i3=-1;i3<=1;i3++)
			//~ {val=d.get(d.vecini(i,i1,i2,i3));
			//~ av+=val;
				//~ if(val<min)min=val;
			//~ }
			
			int x=0;
			for(int i1=-1;i1<=1;i1++)
			for(int i2=-1;i2<=1;i2++)
			for(int i3=-1;i3<=1;i3++){
				val=d.get(d.vecini(i,i1,i2,i3));
				x++;
				if(val>max)max=val;
			}
			
			c.set(i,max);
		

}
}
}

void dilateGrayscaleGeneral(cubee &c,int n){
	for(int q=0;q<n;q++){
		cubee d=c;
		vector<double> ad={0.5,0.6,0.5,0.6,0.8,0.6,0.5,0.6,0.5,0.6,0.8,0.6,0.8,1,0.8,0.6,0.8,0.6,0.5,0.6,0.5,0.6,0.8,0.6,0.5,0.6,0.5};
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double max=-10000;
			double val;
			
			//~ for(int i1=-1;i1<=1;i1++)
			//~ for(int i2=-1;i2<=1;i2++)
			//~ for(int i3=-1;i3<=1;i3++)
			//~ {val=d.get(d.vecini(i,i1,i2,i3));
			//~ av+=val;
				//~ if(val<min)min=val;
			//~ }
			
			int x=0;
			for(int i1=-1;i1<=1;i1++)
			for(int i2=-1;i2<=1;i2++)
			for(int i3=-1;i3<=1;i3++){
				val=d.get(d.vecini(i,i1,i2,i3))*ad[x];
				x++;
				if(val>max)max=val;
			}
			
			c.set(i,max);
		

}
}
}

void blurGrayscale(cubee &c,int n){
	
	for(int q=0;q<n;q++){
		cubee d=c;
			
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double max=-10000;
			double val;
			
			for(int i1=-1;i1<=1;i1++)
			for(int i2=-1;i2<=1;i2++)
			for(int i3=-1;i3<=1;i3++)
			{val=d.get(d.vecini(i,i1,i2,i3));
			av+=val;
				if(val>max)max=val;
			}
			
			//~ for(int k=-1;k<=1;k++){
				//~ val=d.get(d.vecini(i,k,0,0));
				//~ if(val>max)max=val; av+=val;
			//~ }
			//~ for(int k=-1;k<=1;k++){
				//~ val=d.get(d.vecini(i,0,k,0));
				//~ if(val>max)max=val; av+=val;
			//~ }
			//~ for(int k=-1;k<=1;k++){
				//~ val=d.get(d.vecini(i,0,0,k));
				//~ if(val>max)max=val; av+=val;
			//~ }
			c.set(i,av/27);
		}
	}
}

cubee medianFilter(cubee &c){
	cubee c1(c.Nx,c.Ny,c.Nz);
		
	for(int i=0;i<c.N_tot;i++){
		double medie=0;
		for(int i1=-1;i1<=1;i1++)
		for(int i2=-1;i2<=1;i2++)
		for(int i3=-1;i3<=1;i3++){
			medie+=c.get(c.vecini(i,i1,i2,i3));
		}
		
		c1.set(i,medie/27.);
	}	
	return c1;
}

bool isOk(int i,int n){
	return i>0&& i<n;
}

cubee distanceMapChessEfficient(cubee const &c,int culoare){
	cubee d=c;
	
	for(int i=0;i<d.N_tot;i++){
		if(d.get(i)==culoare){
			d.set(i,d.Nx*d.Ny);
		}
	}
	
	float a0,a1,a2,a3;
	int j1,j2,j3;
	vector<float>v;
	for(int i1=0;i1<d.Nx;i1++)
	for(int i2=0;i2<d.Ny;i2++)
	for(int i3=0;i3<d.Nz;i3++){
		a0=d.get(i1,i2,i3);
		
		a1=0;
		j1=i1-1;
		j2=i2;
		j3=i3;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a1=d.get(j1,j2,j3);
		}
		
		a2=0;
		j1=i1;
		j2=i2-1;
		j3=i3;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a2=d.get(j1,j2,j3);
		}
		
		a3=0;
		j1=i1;
		j2=i2;
		j3=i3-1;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a3=d.get(j1,j2,j3);
		}
		v={a0,a1+1,a2+1,a3+1};
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
	}
	
	for(int i1=d.Nx-1;i1>=0;i1--)
	for(int i2=d.Ny-1;i2>=0;i2--)
	for(int i3=d.Nz-1;i3>=0;i3--){
		a0=d.get(i1,i2,i3);
		
		a1=0;
		j1=i1+1;
		j2=i2;
		j3=i3;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a1=d.get(j1,j2,j3);
		}
		
		a2=0;
		j1=i1;
		j2=i2+1;
		j3=i3;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a2=d.get(j1,j2,j3);
		}
		
		a3=0;
		j1=i1;
		j2=i2;
		j3=i3+1;
		
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a3=d.get(j1,j2,j3);
		}
	
		v={a0,a1+1,a2+1,a3+1};
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
		}
	
	return d;
	
	


}

cubee distanceMapChebEfficient(cubee const &c,int culoare){
	cubee d=c;
	
	for(int i=0;i<d.N_tot;i++){
		if(d.get(i)==culoare){
			d.set(i,d.Nx*d.Ny);
		}
	}
	
	float a0;
	int j1,j2,j3;
	vector<float>v;
	
	for(int i1=0;i1<d.Nx;i1++)
	for(int i2=0;i2<d.Ny;i2++)
	for(int i3=0;i3<d.Nz;i3++){
		a0=d.get(i1,i2,i3);
		v.push_back(a0);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
	}
	cout<<v.size()<<endl;
	for(int i1=d.Nx-1;i1>=0;i1--)
	for(int i2=d.Ny-1;i2>=0;i2--)
	for(int i3=d.Nz-1;i3>=0;i3--){
		a0=d.get(i1,i2,i3);
		v.push_back(a0);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		//***
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
		a0=0;
		j1=i1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+1);
		
			
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
		}
	
	return d;
	
	


}

cubee distanceMap(cubee const &c,int culoare){
	cubee dd(c.Nx,c.Ny,c.Nz);	
	cubee old=c;
	cubee curent=c;	
	double sum=volume(curent,culoare);
	int step=0;
	while(sum>0){
		step++;
		old=curent;
		//~ erode(curent,culoare,1);
		erodeGrayscale(curent,1);
		for(int i=0;i<curent.N_tot;i++){
			if(curent.get(i)-old.get(i)<-0.1){
				dd.set(i,step);
			}
		
		}
		
		sum=volume(curent,culoare);
	}
	
	return dd;
		

}


cubee distanceMapGeneralEfficient(cubee const &c,int culoare){
	cubee d=c;
	
	for(int i=0;i<d.N_tot;i++){
		if(d.get(i)==culoare){
			d.set(i,d.Nx*d.Ny);
		}
	}
	
	float a0;
	int j1,j2,j3;
	vector<float>v;
	vector<float>adaugat={5,4,5,4,3,4,5,4,5,4,3,4,3};
	//vector<float>adaugat={3,2,3,2,1,2,3,2,3,2,1,2,1};//chessBoard
	float r3=sqrt(3);
	float r2=sqrt(2);
	//vector<float>adaugat={r3,r2,r3,r2,1,r2,r3,r2,r3,r2,1,r2,1};
	for(int i1=0;i1<d.Nx;i1++)
	for(int i2=0;i2<d.Ny;i2++)
	for(int i3=0;i3<d.Nz;i3++){
		a0=d.get(i1,i2,i3);
		v.push_back(a0);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[0]);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[1]);
		
		a0=0;
		j1=i1-1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[2]);
		//***
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[3]);
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[4]);
		
		a0=0;
		j1=i1-1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[5]);
		//***
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[6]);
		
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[7]);
		
		a0=0;
		j1=i1-1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[8]);
		//***
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[9]);
		
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[10]);
		
		a0=0;
		j1=i1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[11]);
		
		a0=0;
		j1=i1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[12]);
		
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
	}
	cout<<v.size()<<endl;
	for(int i1=d.Nx-1;i1>=0;i1--)
	for(int i2=d.Ny-1;i2>=0;i2--)
	for(int i3=d.Nz-1;i3>=0;i3--){
		a0=d.get(i1,i2,i3);
		v.push_back(a0);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[0]);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[1]);
		
		a0=0;
		j1=i1+1;
		j2=i2-1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[2]);
		//***
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[3]);
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[4]);
		
		a0=0;
		j1=i1+1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[5]);
		//***
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[6]);
		
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[7]);
		
		a0=0;
		j1=i1+1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[8]);
		//***
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3-1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[9]);
		
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[10]);
		
		a0=0;
		j1=i1;
		j2=i2+1;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[11]);
		
		a0=0;
		j1=i1;
		j2=i2;
		j3=i3+1;
		if(isOk(d.index(j1,j2,j3),d.N_tot)){
			a0=d.get(j1,j2,j3);
		}
		v.push_back(a0+adaugat[12]);
		
			
		d.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
		}
	
	return d;
}

cubee distanceMapGeneralEfficientMare(cubee const &cu,int culoare){
	cubee d1=cu;
	
	for(int i=0;i<d1.N_tot;i++){
		if(d1.get(i)==culoare){
			d1.set(i,d1.Nx*d1.Ny);
		}
	}
	
	float a0;
	int j1,j2,j3;
	vector<float>v;
	float a,b,c,d,e,f;
	a=23;
	b=32;
	c=39;
	d=51;
	e=55;
	f=68;
	vector<int> in;


	for(int i1=0;i1<d1.Nx;i1++)
	for(int i2=0;i2<d1.Ny;i2++)
	for(int i3=0;i3<d1.Nz;i3++){
		a0=d1.get(i1,i2,i3);
		v.push_back(a0);
		
		//-2
		in={-2,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-2,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-2,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-2,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		
		in={-2,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-2,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-2,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-2,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		
		in={-2,1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,-1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,-1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,-2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,-2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-2,2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		
		//-1
		in={-1,0,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		
		in={-1,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		
		in={-1,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		
		in={-1,2,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-1,-2,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-1,0,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={-1,0,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		
		in={-1,-2,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-1,-2,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-1,2,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={-1,2,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		
		in={-1,-2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,-2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,-1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,-1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={-1,2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		
		//0;
		in={0,-2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,-2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,-1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,-1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
				
		d1.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
	}
	cout<<v.size()<<endl;
	for(int i1=d1.Nx-1;i1>=0;i1--)
	for(int i2=d1.Ny-1;i2>=0;i2--)
	for(int i3=d1.Nz-1;i3>=0;i3--){
		a0=d1.get(i1,i2,i3);
		v.push_back(a0);
		
		//0
		in={0,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={0,2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
			

		//1
		in={1,0,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		
		in={1,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		
		in={1,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		
		in={1,2,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={1,-2,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={1,0,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={1,0,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		
		in={1,-2,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={1,-2,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={1,2,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={1,2,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		
		in={1,-2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,-2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,-1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,-1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={1,2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		
		//2
		in={2,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={2,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={2,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		in={2,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+d);
		
		in={2,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={2,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={2,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		in={2,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+e);
		
		in={2,1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,-1,2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,-1,-2};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,-2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,-2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,2,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		in={2,2,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+f);
		
		d1.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
		}
	
	return d1;
}

cubee distanceMapGeneralEfficientMic(cubee const &cu,int culoare){
	cubee d1=cu;
	
	for(int i=0;i<d1.N_tot;i++){
		if(d1.get(i)==culoare){
			d1.set(i,d1.Nx*d1.Ny);
		}
	}
	
	float a0;
	int j1,j2,j3;
	vector<float>v;
	float a,b,c;
	a=3;
	b=4;
	c=5;
	vector<int> in;
	for(int i1=0;i1<d1.Nx;i1++)
	for(int i2=0;i2<d1.Ny;i2++)
	for(int i3=0;i3<d1.Nz;i3++){
		a0=d1.get(i1,i2,i3);
		v.push_back(a0);		
		//-1
		in={-1,0,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);		
		in={-1,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={-1,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);		
		in={-1,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={-1,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
				
		//0;	
		in={0,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
				
		d1.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
	}
	cout<<v.size()<<endl;
	for(int i1=d1.Nx-1;i1>=0;i1--)
	for(int i2=d1.Ny-1;i2>=0;i2--)
	for(int i3=d1.Nz-1;i3>=0;i3--){
		a0=d1.get(i1,i2,i3);
		v.push_back(a0);
		
		//0
		in={0,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={0,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);
		in={0,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		//1
		in={1,0,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+a);		
		in={1,1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,-1,0};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,0,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);
		in={1,0,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+b);		
		in={1,1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,-1,1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
		in={1,-1,-1};v.push_back(d1.getSafe(i1+in[0],i2+in[1],i3+in[2])+c);
	
		d1.set(i1,i2,i3,*min_element(v.begin(),v.end()));
		v.clear();
		}
	
	return d1;
}


cubee distanceMapCheb(cubee const &c,int culoare){
	cubee dd(c.Nx,c.Ny,c.Nz);
		
	cubee old=c;
	cubee curent=c;	
	double sum=volume(curent,culoare);
	int step=0;
	while(sum>0){
		step++;
		old=curent;
		//~ erode(curent,culoare,1);
		erodeGrayscaleCheb(curent,1);
		for(int i=0;i<curent.N_tot;i++){
			if(curent.get(i)-old.get(i)<-0.1){
				dd.set(i,step);
			}		
		}		
		sum=volume(curent,culoare);
	}
	
	return dd;
}


cubee morphologicalReconstruction(cubee marker,cubee &mask){
	
	cubee init=marker;
	for(int q=0;q<50;q++){
		cubee c=init;
		//cout<<volume(c)<<endl;
		for(int i=0;i<c.N_tot;i++){
			double av=0;
			double max=-10000;
			double val;
			
			for(int k=-1;k<=1;k++){
				val=c.get(c.vecini(i,k,0,0));
				if(val>max)max=val;
			}
			for(int k=-1;k<=1;k++){
				val=c.get(c.vecini(i,0,k,0));
				if(val>max)max=val;
			}
			for(int k=-1;k<=1;k++){
				val=c.get(c.vecini(i,0,0,k));
				if(val>max)max=val;
			}
			if(max<=mask.get(i))
			{
				//cout<<max<<endl;
				init.set(i,max);
			}
		}
	}	
	return init;
}


cubee morphologicalReconstructionHybrid(cubee marker,cubee mask){
	for(int i1=0;i1<marker.Nx;i1++)
	for(int i2=0;i2<marker.Ny;i2++)
	for(int i3=0;i3<marker.Nz;i3++){
		int i=marker.index(i1,i2,i3);
		
		double max=marker.get(i);
		
		int iVecin=marker.vecini(i,-1,0,0);
		double val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		iVecin=marker.vecini(i,1,0,0);
		val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		iVecin=marker.vecini(i,0,-1,0);
		val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		iVecin=marker.vecini(i,0,1,0);
		val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		iVecin=marker.vecini(i,0,0,-1);
		val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		iVecin=marker.vecini(i,0,0,1);
		val=marker.get(iVecin);
		if(iVecin<=i){
			if(val>max){
				max=val;
			}		
		}
		
		if(max<=mask.get(i)){
			//cout<<max<<endl;
			marker.set(i,max);
		}
		else{
			marker.set(i,mask.get(i));
		}
	}
	
	deque<int> fifo;
	for(int i3=0;i3<marker.Nz;i3++)
	for(int i2=0;i2<marker.Ny;i2++)
	for(int i1=0;i1<marker.Nx;i1++){
		int i=marker.index(i1,i2,i3);
		
		 double max1=marker.get(i);
		
	     double iVecin=marker.vecini(i,-1,0,0);
		double val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		iVecin=marker.vecini(i,1,0,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		iVecin=marker.vecini(i,0,-1,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		iVecin=marker.vecini(i,0,1,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		iVecin=marker.vecini(i,0,0,-1);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		iVecin=marker.vecini(i,0,0,1);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val>max1){
				max1=val;
			}		
		}
		
		if(max1<=mask.get(i))
			{
				//cout<<max<<endl;
				marker.set(i,max1);
			}
			else{
				marker.set(i,mask.get(i));
			}
	
		bool exista=false;
		
		double val0=marker.get(i);
		double maskVal=mask.get(i);
	
		iVecin=marker.vecini(i,-1,0,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}iVecin=marker.vecini(i,1,0,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}iVecin=marker.vecini(i,0,-1,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}iVecin=marker.vecini(i,0,1,0);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}iVecin=marker.vecini(i,0,0,-1);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}iVecin=marker.vecini(i,0,0,1);
		val=marker.get(iVecin);
		if(iVecin>=i){
			if(val<val0 && val<maskVal){
				exista=true;
				
			}		
		}
		
		if(exista){
			fifo.push_back(i);
		
		}
	
	}
	
	while(fifo.size()>0){
		//cout<<fifo.size()<<endl;
		int i=fifo.front();
		fifo.pop_front();
	
		int iVecin=marker.vecini(i,-1,0,0);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
		iVecin=marker.vecini(i,1,0,0);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
		iVecin=marker.vecini(i,0,-1,0);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
		iVecin=marker.vecini(i,0,1,0);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
		 iVecin=marker.vecini(i,0,0,-1);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
		
		iVecin=marker.vecini(i,0,0,1);
		if(marker.get(iVecin)<marker.get(i) && mask.get(iVecin)!=marker.get(iVecin)){
			double minim;
			if(marker.get(i)<mask.get(iVecin)){
				minim=marker.get(i);
			}else{
				minim=mask.get(iVecin);
			}
			marker.set(iVecin,minim);
			fifo.push_back(iVecin);
		}
	}
	
	
	return marker;
	
	
	}


struct thing
{
    int pixel;
    float valoare;
	bool operator<(const thing& rhs) const{
		return valoare < rhs.valoare;
	}
	
	bool operator>(const thing& rhs) const{
		return valoare > rhs.valoare;
	}

	bool operator<=(const thing& rhs) const{
		return valoare <= rhs.valoare;
	}

	bool operator>=(const thing& rhs) const	{
		return valoare >= rhs.valoare;
	}
};


cubee priorityFlood(cubee d,cubee seed){
	//trebuie sa fie distance map
	
	priority_queue<thing> open;
	cubee closed(d.Nx,d.Ny,d.Nz);
	//cubee filled(d.Nx,d.Ny,d.Nz)
	
	for(int i=0;i<seed.N_tot;i++){
		if(seed.get(i)>0){
			open.push(thing{i,seed.get(i)});
			closed.set(i,seed.get(i));
			//filled.set(i,seed.get(i));
		}
	}
	
	while(open.size()>0){
		thing curent=open.top();
		open.pop();
		vector<int> vecini;
		
		int i=curent.pixel;
		int iVecin=d.vecini(i,-1,0,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,1,0,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,-1,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
			}
		iVecin=d.vecini(i,0,1,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,-1);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,1);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}

		for(int k=0;k<vecini.size();k++){
			iVecin=vecini[k];
			if(closed.get(iVecin)==0){
				d.set(iVecin,std::min(d.get(iVecin),d.get(i)));
				closed.set(iVecin,closed.get(i));
				open.push(thing{iVecin,d.get(iVecin)});
			}
		}
	}
	return closed;
	
	}
	
cubee priorityFloodModificat(cubee d,cubee seed){
	//trebuie sa fie distance map
	
	priority_queue<thing> open;
	cubee closed(d.Nx,d.Ny,d.Nz);
	//cubee filled(d.Nx,d.Ny,d.Nz)
	
	for(int i=0;i<seed.N_tot;i++){
		if(seed.get(i)>0){
			open.push(thing{i,seed.get(i)});
			closed.set(i,seed.get(i));
			//filled.set(i,seed.get(i));
		}
	}
	
	while(open.size()>0){
		thing curent=open.top();
		open.pop();
		vector<int> vecini;
		
		int i=curent.pixel;
		int iVecin=d.vecini(i,-1,0,0);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,1,0,0);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,-1,0);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
			}
		iVecin=d.vecini(i,0,1,0);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,-1);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,1);
		if(d.get(iVecin)==0){
			vecini.push_back(iVecin);
		}

		for(int k=0;k<vecini.size();k++){
			iVecin=vecini[k];
			if(closed.get(iVecin)!=0){
				d.set(iVecin,std::min(d.get(iVecin),d.get(i)));
				closed.set(iVecin,closed.get(i));
				open.push(thing{iVecin,d.get(iVecin)});
			}
		}
	}
	return closed;
	
	}
	
	

cubee priorityFlood(cubee d){
	//trebuie sa fie distance map
	
	
	priority_queue<thing> open;
	cubee closed(d.Nx,d.Ny,d.Nz);
	
	open.push(thing{d.index(50,50,50),d.get(50,50,50)});
	open.push(thing{d.index(100,50,50),d.get(100,50,50)});
	open.push(thing{d.index(100,50,50),d.get(100,50,55)});
	open.push(thing{d.index(100,100,50),d.get(100,100,50)});
	open.push(thing{d.index(100,75,100),d.get(100,75,100)});
	closed.set(50,50,50,1);
	closed.set(100,50,50,2);
	closed.set(100,50,55,2.5);
	closed.set(100,100,50,3);
	closed.set(100,75,100,4);
	
	while(open.size()>0){
		thing curent=open.top();
		//cout<<d.index3D(curent.pixel)[0]<<" "<<d.index3D(curent.pixel)[1]<<" "<<d.index3D(curent.pixel)[2]<<" "<<d.get(curent.pixel)<<" "<<curent.valoare<<endl;
		
		open.pop();
		vector<int> vecini;
		
		int i=curent.pixel;
		int iVecin=d.vecini(i,-1,0,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,1,0,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,-1,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
			}
		iVecin=d.vecini(i,0,1,0);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,-1);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		iVecin=d.vecini(i,0,0,1);
		if(d.get(iVecin)!=0){
			vecini.push_back(iVecin);
		}
		

		for(int k=0;k<vecini.size();k++){
			iVecin=vecini[k];
			if(closed.get(iVecin)==0){
				d.set(iVecin,std::min(d.get(iVecin),d.get(i)));
				closed.set(iVecin,closed.get(i));
				open.push(thing{iVecin,d.get(iVecin)});
				
			}
		
		}
		
		
	}
	return closed;
	
	}
	
	
		

void otsu(cubee& c){
	double max = *max_element(c.Val.begin(), c.Val.end());
	double min = *min_element(c.Val.begin(), c.Val.end());
	int nn=300;
	double dt=(max-min)/(nn-1);
	
	vector<float> hist(nn);
	
	for(int i=0;i<c.N_tot;i++){
		hist[(int)((c.get(i)-min)/dt)]++;
	}
	
	ofstream otsuF("otsu");
	
	for(int t=1;t<nn-1;t++){
	double w0=0;
	double w1=0;
	double m0=0;
	double m1=0;
		double th=min+t*dt;
		
		for(int i=0;i<t;i++){
			w0+=hist[i];
		}
		for(int i=t;i<nn;i++){
			w1+=hist[i];
		}
		
		for(int i=0;i<t;i++){
			m0+=hist[i]*i/w0;
		}
		for(int i=t;i<nn;i++){
			m1+=hist[i]*i/w1;
		}		
		otsuF<<th<<" "<<w0*w1*pow(m0-m1,2)<<endl;
	}
}


vector<float> otsu(cubee& c,float xMin,float xMax,int n){
	double min =xMin; 
	double max =xMax;
	int nn=n;
	double dt=(max-min)/(nn-1);
	
	vector<float> hist(nn);
	
	for(int i=0;i<c.N_tot;i++){
		hist[(int)((c.get(i)-min)/dt)]++;
	}
	
	vector<float> otsu;
	
	for(int t=1;t<nn-1;t++){
	double w0=0;
	double w1=0;
	double m0=0;
	double m1=0;
		double th=min+t*dt;
		
		for(int i=0;i<t;i++){
			w0+=hist[i];
		}
		for(int i=t;i<nn;i++){
			w1+=hist[i];
		}
		
		for(int i=0;i<t;i++){
			m0+=hist[i]*i/w0;
		}
		for(int i=t;i<nn;i++){
			m1+=hist[i]*i/w1;
		}		
		otsu.push_back(w0*w1*pow(m0-m1,2));
	}
	
	return otsu;
}


template<typename T> void print_queue(T q) {
    while(!q.empty()) {
        std::cout << q.top().pixel << " "<<q.top().valoare<<" "<<endl;
        q.pop();
    }
    std::cout << '\n';
}


void addSphere(cubee& c, float x,float y,float z, float raza,float smo=0){
	if(smo==0){
		for(int i=0;i<c.Nx;i++)
		for(int j=0;j<c.Ny;j++)
		for(int k=0;k<c.Nz;k++){
		double r=sqrt(pow(i-x,2)+pow(j-y,2)+pow(k-z,2));
		if(r<=raza){
			c.set(i,j,k,1);
		}	
			
			
		}
	}else{
		for(int i=0;i<c.Nx;i++)
		for(int j=0;j<c.Ny;j++)
		for(int k=0;k<c.Nz;k++){
		double r=sqrt(pow(i-x,2)+pow(j-y,2)+pow(k-z,2));
		
		double v=1./(1+exp((r-raza)/smo));
		
		double val_init=c.get(i,j,k);
		if(val_init+v<=1){
			c.set(i,j,k,val_init+v);
		}else{
			c.set(i,j,k,1);
		}
	
	
		}
	}
	
	
}



Eigen::Vector3d euler_angles(Eigen::Matrix3d rot){
   //(Z,Y,Z)
    Eigen::Vector3d v;
    double alfa;
    double beta;
    double gamma;
   
    //~ if(rot(2,2)!=1&&rot(2,2)!=-1){
        //~ beta=acos(rot(2,2));
        //~ alfa=atan2(rot(1,2),rot(0,2));
        //~ gamma=atan2(rot(2,1),-rot(2,0));
    //~ }else{
        //~ if(rot(2,2)==1){
            //~ beta=0;
            //~ gamma=0;
            //~ alfa=atan2(rot(1,0),rot(0,0));
            
        //~ }
        
        //~ if(rot(2,2)==-1){
            //~ beta=pi;
            //~ gamma=0;
            //~ alfa=-atan2(rot(1,0),rot(1,1));
            
        //~ }

        
    //~ }
    
	if(rot(2,2)!=1&&rot(2,2)!=-1){
        alfa=atan2(rot(1,2),rot(0,2));
        gamma=atan2(rot(2,1),-rot(2,0));
        beta=atan2(rot(0,2)*cos(alfa)+rot(1,2)*sin(alfa),rot(2,2));
        
    }else{
        if(rot(2,2)==1){
            beta=0;
            gamma=0;
            alfa=atan2(-rot(0,1),rot(0,0));
            
        }
        
        if(rot(2,2)==-1){
			//cout<<"asd"<<endl;
			//cout<<rot<<endl;//asta nu merge bine -> am rezolvat 
            beta=pi;
            gamma=0;
            alfa=atan2(-rot(0,1),-rot(0,0));//trebuie sa fie semnul exact
            
        }

        
    }
    
    v(0)=alfa;
    v(1)=beta;
    v(2)=gamma;
    return v;
    
}

Eigen::Vector3d breit_angles(Eigen::Matrix3d rot){
   //(Z,Y,Z)
    Eigen::Vector3d v;
    double alfa;
    double beta;
    double gamma;
   
	alfa=atan2(rot(1,0),rot(0,0));
	gamma=atan2(rot(2,1),rot(2,2));
	beta=-asin(rot(2,0));

    v(0)=alfa;
    v(1)=beta;
    v(2)=gamma;
    return v;
    
}
Eigen::Matrix3d euler_matrix(double alfa,double beta,double gamma){
    //(Z,Y,Z)
    
    Eigen::Matrix3d A,B,C;
    A<<cos(alfa),-sin(alfa),0,
        sin(alfa),cos(alfa),0,
        0,0,1;
        
    B<<cos(beta),0,sin(beta),
        0,1,0,
        -sin(beta),0,cos(beta);
        
    C<<cos(gamma),-sin(gamma),0,
        sin(gamma),cos(gamma),0,
        0,0,1;
    
    return A*B*C;
    
}

Eigen::Matrix3d breit_matrix(double alfa,double beta,double gamma){
    //(Z,Y,X)
    
    Eigen::Matrix3d A,B,C;
    A<<cos(alfa),-sin(alfa),0,
        sin(alfa),cos(alfa),0,
        0,0,1;
        
    B<<cos(beta),0,sin(beta),
        0,1,0,
        -sin(beta),0,cos(beta);
        
    C<<1,0,0,
    0,cos(gamma),-sin(gamma),
    0,sin(gamma),cos(gamma);
       
    return C*B*A;
    
}




void addEllipse(cubee& cc, float x1,float x2,float x3,float a1,float a2,float a3,float alfa,float beta,float gamma,float scala){
	
		Eigen::Matrix3d m=euler_matrix(alfa,beta,gamma);
		//~ cout<<"euler"<<endl;
		//~ cout<<m<<endl;
		for(int i1=0;i1<cc.Nx;i1++)
		for(int i2=0;i2<cc.Ny;i2++)
		for(int i3=0;i3<cc.Nz;i3++){
		
		float x=i1-x1,y=i2-x2,z=i3-x3;
		Vec3 vv={x,y,z};
		vv=m.transpose()*vv;
		//~ vv=m*vv;
		double w=pow(vv[0]/a1,2)+pow(vv[1]/a2,2)+pow(vv[2]/a3,2)-scala;
		float v=1./(1+exp(w*10));
		
		
		double val_init=cc.get(i1,i2,i3);
		if(val_init+v<=1){
			cc.set(i1,i2,i3,val_init+v);
		}else{
			cc.set(i1,i2,i3,1);
		}
		}
}

void addEllipse2(cubee& cc, float x1,float x2,float x3,float a1,float a2,float a3,Eigen::Matrix3d m,float scala,int color=1){
	
		//~ Eigen::Matrix3d m=breit_matrix(alfa,beta,gamma);
		for(int i1=0;i1<cc.Nx;i1++)
		for(int i2=0;i2<cc.Ny;i2++)
		for(int i3=0;i3<cc.Nz;i3++){
		
		float x=i1-x1,y=i2-x2,z=i3-x3;
		Vec3 vv={x,y,z};
		vv=m.transpose()*vv;
		
		double w=pow(vv[0]/a1,2)+pow(vv[1]/a2,2)+pow(vv[2]/a3,2)-scala;
		float v=color/(1.+exp(w*10));
		
		
		double val_init=cc.get(i1,i2,i3);
		if(val_init+v<=1){
			cc.set(i1,i2,i3,val_init+v);
		}else{
			cc.set(i1,i2,i3,color);
		}
	
	
		}
	
	
	
}

void addEllipse3(cubee& cc, float x1,float x2,float x3,float a1,float a2,float a3,float alfa,float beta,float gamma,float scala){
	
		Eigen::Matrix3d m=euler_matrix(alfa,beta,gamma);
		
		float maxim=0;
		if(x1>maxim)maxim=a1;
		if(x2>maxim)maxim=a2;
		if(x3>maxim)maxim=a3;
		
		int l=(int)(1.2*maxim);
		int i1C=(int)x1;
		int i2C=(int)x2;
		int i3C=(int)x3;
		
		for(int i1=-l;i1<l;i1++)
		for(int i2=-l;i2<l;i2++)
		for(int i3=-l;i3<l;i3++){
		
			
			float x=i1+i1C-x1,y=i2+i2C-x2,z=i3+i3C-x3;
			
			Vec3 vv={x,y,z};
			//~ cout<<x<<" "<<y<<" "<<z<<endl;
			vv=m.transpose()*vv;
		
			double w=pow(vv[0]/a1,2)+pow(vv[1]/a2,2)+pow(vv[2]/a3,2)-scala;
			float v=1./(1.+exp(w*10));
		
		
			double val_init=cc.getSafe(i1+i1C,i2+i2C,i3+i3C);
			if(val_init+v<=1){
				cc.setSafe(i1+i1C,i2+i2C,i3+i3C,val_init+v);
			}else{
				cc.setSafe(i1+i1C,i2+i2C,i3+i3C,1);
			}
	
	
		}

}



vector<double> to_vec1(string str){
stringstream iss(str);
	double number;
	std::vector<double> myNumbers;
	while ( iss >> number )
	  myNumbers.push_back( number );
	return myNumbers;    

}

vector<vector<double>> read_fis(string nume_fila){
    string str; 
    vector<vector<double>> vv;
    vector<double> v;
    ifstream fil(nume_fila.c_str());
    while (getline(fil, str)){
        if(str.size() > 0){
            v=to_vec1(str);
        }
        vv.push_back(v);
    }
    
    return vv;
    
}


void addPolygon(cubee& c){
	vector<vector<double>> v=read_fis("fisier_poli.dat");
	
	int nrPuncte=(int)v[0][0];
	int nrFete=(int)v[0][1];
	
	vector<Vec3> puncte;
	for(int i=1;i<1+nrPuncte;i++){
		Vec3 v1;
		v1<<v[i][0],v[i][1],v[i][2];
		puncte.push_back(v1);	
	}
	
	vector<vector<int>> fete;
	vector<Vec3> normale;
	
	for(int i=1+nrPuncte;i<1+nrPuncte+nrFete;i++){
		vector<int> v1;
		for(int j=0;j<v[i].size();j++){
			v1.push_back((int)v[i][j]-1);
		}
		fete.push_back(v1);
	
	}
	
	for(int i=0;i<nrFete;i++){
		Vec3 no;
		no=(puncte[fete[i][1]]-puncte[fete[i][0]]).cross(puncte[fete[i][2]]-puncte[fete[i][1]]);
		no.normalize();
		normale.push_back(no);	
	}
	
	
	Vec3 cm;
	cm<<0,0,0;
	for(int i=0;i<nrPuncte;i++){
		cm+=puncte[i];
	}
	cm=cm/nrPuncte;
	
	
	
	double lx,ly,lz;
	lx=0,ly=0,lz=0;
	
	for(int i=0;i<nrPuncte;i++){
		double x,y,z;
		x=puncte[i][0];
		y=puncte[i][1];
		z=puncte[i][2];
		
		if(fabs(x-cm[0])>lx){lx=fabs(x-cm[0]);}
		if(fabs(y-cm[1])>ly){ly=fabs(y-cm[1]);}
		if(fabs(z-cm[2])>lz){lz=fabs(z-cm[2]);}
	
	}

		int lx1=(int)(1.2*lx),ly1=(int)(1.2*ly),lz1=(int)(1.2*lz);
		int i1C=(int)cm[0];
		int i2C=(int)cm[1];
		int i3C=(int)cm[2];
		//~ cout<<lx1<<" "<<ly1<<" "<<lz1<<endl;
		//~ cout<<i1C<<" "<<i2C<<" "<<i3C<<endl;
		for(int i1=-lx1;i1<lx1;i1++)
		for(int i2=-ly1;i2<ly1;i2++)
		for(int i3=-lz1;i3<lz1;i3++){
			c.set(i1+i1C,i2+i2C,i3+i3C,1);
		}
	
		for(int i1=-lx1;i1<lx1;i1++)
		for(int i2=-ly1;i2<ly1;i2++)
		for(int i3=-lz1;i3<lz1;i3++){
		
			for(int i=0;i<nrFete;i++){
				Vec3 r1;
				r1<<i1+i1C,i2+i2C,i3+i3C;
				//~ r1=cm;
				//~ cout<<(r1-puncte[fete[i][1]]).dot(normale[i])<<endl;
				if((r1-puncte[fete[i][1]]).dot(normale[i])>0){
				c.set(i1+i1C,i2+i2C,i3+i3C,0);
				}
			}
		}
}

void addPolygon(cubee& c,string numeFisier,double xc,double yc,double zc,double scala = 1){
	vector<vector<double>> v=read_fis(numeFisier.c_str());
	
	int nrPuncte=(int)v[0][0];
	int nrFete=(int)v[0][1];
	
	vector<Vec3> puncte;
	for(int i=1;i<1+nrPuncte;i++){
		Vec3 v1;
		v1<<scala*v[i][0]+xc,scala*v[i][1]+yc,scala*v[i][2]+zc;
		puncte.push_back(v1);	
	}
	
	vector<vector<int>> fete;
	vector<Vec3> normale;
	
	for(int i=1+nrPuncte;i<1+nrPuncte+nrFete;i++){
		vector<int> v1;
		for(int j=0;j<v[i].size();j++){
			v1.push_back((int)v[i][j]-1);
		}
		fete.push_back(v1);
	
	}
	
	for(int i=0;i<nrFete;i++){
		Vec3 no;
		no=(puncte[fete[i][1]]-puncte[fete[i][0]]).cross(puncte[fete[i][2]]-puncte[fete[i][1]]);
		no.normalize();
		normale.push_back(no);	
	}
	
	
	Vec3 cm;
	cm<<0,0,0;
	for(int i=0;i<nrPuncte;i++){
		cm+=puncte[i];
	}
	cm=cm/nrPuncte;
	
	
	
	double lx,ly,lz;
	lx=0,ly=0,lz=0;
	
	for(int i=0;i<nrPuncte;i++){
		double x,y,z;
		x=puncte[i][0];
		y=puncte[i][1];
		z=puncte[i][2];
		
		if(fabs(x-cm[0])>lx){lx=fabs(x-cm[0]);}
		if(fabs(y-cm[1])>ly){ly=fabs(y-cm[1]);}
		if(fabs(z-cm[2])>lz){lz=fabs(z-cm[2]);}
	
	}

		int lx1=(int)(1.2*lx),ly1=(int)(1.2*ly),lz1=(int)(1.2*lz);
		int i1C=(int)cm[0];
		int i2C=(int)cm[1];
		int i3C=(int)cm[2];
		//~ cout<<lx1<<" "<<ly1<<" "<<lz1<<endl;
		//~ cout<<i1C<<" "<<i2C<<" "<<i3C<<endl;
		for(int i1=-lx1;i1<lx1;i1++)
		for(int i2=-ly1;i2<ly1;i2++)
		for(int i3=-lz1;i3<lz1;i3++){
			c.set(i1+i1C,i2+i2C,i3+i3C,1);
		}
	
		for(int i1=-lx1;i1<lx1;i1++)
		for(int i2=-ly1;i2<ly1;i2++)
		for(int i3=-lz1;i3<lz1;i3++){
		
			for(int i=0;i<nrFete;i++){
				Vec3 r1;
				r1<<i1+i1C,i2+i2C,i3+i3C;
				//~ r1=cm;
				//~ cout<<(r1-puncte[fete[i][1]]).dot(normale[i])<<endl;
				if((r1-puncte[fete[i][1]]).dot(normale[i])>0){
				c.set(i1+i1C,i2+i2C,i3+i3C,0);
				}
			}
		}
}









vector<int> volumeAnalysis(cubee& c){
	
	vector<int> v(32);
	double culoareMax=-1;
	
	
	
	for(int i=0;i<c.N_tot;i++){		
		int culoare=c.get(i);		
		if(culoare>=culoareMax){
			culoareMax=culoare;
			v.resize(culoare+1);
		}	
		v[culoare]+=1;
	}
	
	vector<float> xcm(culoareMax+1);
	vector<float> ycm(culoareMax+1);
	vector<float> zcm(culoareMax+1);
	vector<Eigen::Matrix3d> mVec(culoareMax+1);
	
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		xcm[culoare]+=c.index3D(i)[0];
		ycm[culoare]+=c.index3D(i)[1];
		zcm[culoare]+=c.index3D(i)[2];
	}
	
	for(int i=0;i<=culoareMax;i++){
		if(v[i]>0){
			xcm[i]=xcm[i]/v[i];
			ycm[i]=ycm[i]/v[i];
			zcm[i]=zcm[i]/v[i];
		}
	}
	
	double x,y,z;
	for(int i=0;i<c.N_tot;i++){
			int culoare=c.get(i);
			x=c.index3D(i)[0]-xcm[culoare];
			y=c.index3D(i)[1]-ycm[culoare];
			z=c.index3D(i)[2]-zcm[culoare];
			
			mVec[culoare](0,0)+=x*x;//y*y+z*z;
			mVec[culoare](1,1)+=y*y;//x*x+z*z;
			mVec[culoare](2,2)+=z*z;//x*x+y*y;
			mVec[culoare](0,1)+=x*y;
			mVec[culoare](0,2)+=x*z;
			mVec[culoare](1,2)+=y*z;
			
		
			mVec[culoare](1,0)=mVec[culoare](0,1);
			mVec[culoare](2,0)=mVec[culoare](0,2);
			mVec[culoare](2,1)=mVec[culoare](1,2);	
	}
	
	for(int i=0;i<=culoareMax;i++){
		if(v[i]>0){
			mVec[i]=mVec[i]/v[i];
			
		}
	}
	

	ofstream outFile("particleData.dat");
	outFile<<"Label Volume xCM yCM zCM Ax Ay Az Vx Vy Vz"<<endl;
	for(int i=1;i<=culoareMax;i++){
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(mVec[i]);
		Eigen::MatrixXd eigVectors=es.eigenvectors().real();
		Eigen::VectorXd eigValues=es.eigenvalues();
		
		Eigen::Vector3d axes;
		axes<<sqrt(5*fabs(eigValues(0))),sqrt(5*fabs(eigValues(1))),sqrt(5*fabs(eigValues(2)));
		cout<<eigValues<<endl;
		Eigen::Vector3d v1=eigVectors.col(0);
		Eigen::Vector3d v2=eigVectors.col(1);
		Eigen::Vector3d v3=eigVectors.col(2);
		
		if(v[i]>0){
			//cout<<mVec[i]<<endl;
			//cout<<i<<" "<<v[i]<<" "<<xcm[i]<<" "<<ycm[i]<<" "<<zcm[i]<<" "<<axes[2]<<" "<<axes[1]<<" "<<axes[0]<<" "<<v1[2]<<" "<<v1[1]<<" "<<v1[0]<<endl;
			outFile<<i<<" "<<v[i]<<" "<<xcm[i]<<" "<<ycm[i]<<" "<<zcm[i]<<" "<<axes[2]<<" "<<axes[1]<<" "<<axes[0]<<" "<<v1[2]<<" "<<v1[1]<<" "<<v1[0]<<endl;
			
		}
		
		
	}
	
	
	
	
	
	return v;
	
	
}

//~ vector<vector<double>> volumeAnalysisOrientation(cubee& c,string numeOutput){
	
	//~ vector<int> v(32);
	//~ double culoareMax=-1;
	//~ vector<vector<double>> output;
	
	
	//~ for(int i=0;i<c.N_tot;i++){		
		//~ int culoare=c.get(i);		
		//~ if(culoare>=culoareMax){
			//~ culoareMax=culoare;
			//~ v.resize(culoare+1);
		//~ }	
		//~ v[culoare]+=1;
	//~ }
	
	//~ vector<float> xcm(culoareMax+1);
	//~ vector<float> ycm(culoareMax+1);
	//~ vector<float> zcm(culoareMax+1);
	//~ vector<Eigen::Matrix3d> mVec(culoareMax+1);
	
	//~ for(int i=0;i<c.N_tot;i++){
		//~ int culoare=c.get(i);
		//~ xcm[culoare]+=c.index3D(i)[0];
		//~ ycm[culoare]+=c.index3D(i)[1];
		//~ zcm[culoare]+=c.index3D(i)[2];
	//~ }
	
	//~ for(int i=0;i<=culoareMax;i++){
		//~ if(v[i]>0){
			//~ xcm[i]=xcm[i]/v[i];
			//~ ycm[i]=ycm[i]/v[i];
			//~ zcm[i]=zcm[i]/v[i];
		//~ }
	//~ }
	
	//~ double x,y,z;
	//~ for(int i=0;i<c.N_tot;i++){
			//~ int culoare=c.get(i);
			//~ x=c.index3D(i)[0]-xcm[culoare];
			//~ y=c.index3D(i)[1]-ycm[culoare];
			//~ z=c.index3D(i)[2]-zcm[culoare];
			
			//~ mVec[culoare](0,0)+=x*x;//y*y+z*z;
			//~ mVec[culoare](1,1)+=y*y;//x*x+z*z;
			//~ mVec[culoare](2,2)+=z*z;//x*x+y*y;
			//~ mVec[culoare](0,1)+=x*y;
			//~ mVec[culoare](0,2)+=x*z;
			//~ mVec[culoare](1,2)+=y*z;
			
		
			//~ mVec[culoare](1,0)=mVec[culoare](0,1);
			//~ mVec[culoare](2,0)=mVec[culoare](0,2);
			//~ mVec[culoare](2,1)=mVec[culoare](1,2);	
	//~ }
	
	//~ for(int i=0;i<=culoareMax;i++){
		//~ if(v[i]>0){
			//~ mVec[i]=mVec[i]/v[i];
			
		//~ }
	//~ }
	
	//~ output.resize(culoareMax+1);
	//~ ofstream analiza(numeOutput.c_str());
	//~ analiza<<"Label Volume xCM yCM zCM Ax Ay Az Vx Vy Vz alpha beta gamma"<<endl;
	//~ string numeUnghi=numeOutput+"_angle";
	
	//~ ofstream unghiuri(numeUnghi.c_str());
	//~ for(int i=1;i<=culoareMax;i++){
		//~ Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(mVec[i]);
		//~ Eigen::MatrixXd eigVectors=es.eigenvectors().real();
		//~ Eigen::VectorXd eigValues=es.eigenvalues();
		
		//~ Eigen::Vector3d axes;
		//~ axes<<sqrt(5*eigValues(0)),sqrt(5*eigValues(1)),sqrt(5*eigValues(2));
		
		//~ Eigen::Vector3d v1=eigVectors.col(0);
		//~ Eigen::Vector3d v2=eigVectors.col(1);
		//~ Eigen::Vector3d v3=eigVectors.col(2);
		
		//~ double x1=v3(0),x2=v3(1),x3=v3(2);
		
		//~ Eigen::Vector3d eulerAng=euler_angles(eigVectors);
		//~ Eigen::Vector3d kv;
		//~ kv<<0,0,1;
		//~ unghiuri<<acos(fabs(kv.dot(eigVectors.col(2))))<<endl;
		//~ if(v[i]>0){
			//~ output[i].push_back(i);
			//~ output[i].push_back(v[i]);
			
			//~ output[i].push_back(xcm[i]);
			//~ output[i].push_back(ycm[i]);
			//~ output[i].push_back(zcm[i]);
			
			//~ output[i].push_back(axes[0]);
			//~ output[i].push_back(axes[1]);
			//~ output[i].push_back(axes[2]);
			
			//~ output[i].push_back(x1);
			//~ output[i].push_back(x2);
			//~ output[i].push_back(x3);
			
			//~ output[i].push_back(eulerAng[0]*180/pi);
			//~ output[i].push_back(eulerAng[1]*180/pi);
			//~ output[i].push_back(eulerAng[2]*180/pi);
			
			
			
		//~ }
		
		//~ if(v[i]>0){
		//~ for(int uu=0;uu<output[i].size();uu++){
			//~ cout<<output[i][uu]<<" ";
			//~ analiza<<output[i][uu]<<" ";
		//~ }
		//~ cout<<endl;
		//~ analiza<<endl;
		//~ }
	//~ }
	//~ return output;
//~ }


void volumeAnalysisOrientation(cubee& c,string numeOutput){
	
	vector<int> v(32);
	double culoareMax=-1;
	vector<vector<double>> output;
	
	
	for(int i=0;i<c.N_tot;i++){		
		int culoare=c.get(i);		
		if(culoare>=culoareMax){
			culoareMax=culoare;
			v.resize(culoare+1);
		}	
		v[culoare]+=1;
	}
	
	vector<float> xcm(culoareMax+1);
	vector<float> ycm(culoareMax+1);
	vector<float> zcm(culoareMax+1);
	vector<Eigen::Matrix3d> mVec(culoareMax+1);
	
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		xcm[culoare]+=c.index3D(i)[0];
		ycm[culoare]+=c.index3D(i)[1];
		zcm[culoare]+=c.index3D(i)[2];
	}
	
	for(int i=0;i<=culoareMax;i++){
		if(v[i]>0){
			xcm[i]=xcm[i]/v[i];
			ycm[i]=ycm[i]/v[i];
			zcm[i]=zcm[i]/v[i];
		}
		
		Eigen::Matrix3d m;
		m<<0,0,0,0,0,0,0,0,0;
		mVec[i]=m;
	}
	
	double x,y,z;
	for(int i=0;i<c.N_tot;i++){
			int culoare=c.get(i);
			x=c.index3D(i)[0]-xcm[culoare];
			y=c.index3D(i)[1]-ycm[culoare];
			z=c.index3D(i)[2]-zcm[culoare];
			
			
			
			mVec[culoare](0,0)+=x*x;//y*y+z*z;
			mVec[culoare](1,1)+=y*y;//x*x+z*z;
			mVec[culoare](2,2)+=z*z;//x*x+y*y;
			mVec[culoare](0,1)+=x*y;
			mVec[culoare](0,2)+=x*z;
			mVec[culoare](1,2)+=y*z;
			
		
			mVec[culoare](1,0)=mVec[culoare](0,1);
			mVec[culoare](2,0)=mVec[culoare](0,2);
			mVec[culoare](2,1)=mVec[culoare](1,2);	
	
	
	}
	
	for(int i=0;i<=culoareMax;i++){
		if(v[i]>0){
			mVec[i]=mVec[i]/v[i];
			
		}
	}
	
	output.resize(culoareMax+1);
	ofstream analiza(numeOutput.c_str());
	analiza<<"Label Volume xCM yCM zCM Ax Ay Az Vx Vy Vz alpha beta gamma"<<endl;
	string numeUnghi=numeOutput+"_angle";
	
	ofstream unghiuri(numeUnghi.c_str());
	for(int i=11;i<=culoareMax;i++){
		//~ cout<<mVec[i]<<endl;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(mVec[i]);
		Eigen::MatrixXd eigVectors=es.eigenvectors().real();
		Eigen::VectorXd eigValues=es.eigenvalues();
		
		Eigen::Vector3d axes;
		axes<<sqrt(fabs(5*eigValues(0))),sqrt(fabs(5*eigValues(1))),sqrt(fabs(5*eigValues(2)));
		
		Eigen::Vector3d v1=eigVectors.col(0);
		Eigen::Vector3d v2=eigVectors.col(1);
		Eigen::Vector3d v3=eigVectors.col(2);
		
		double x1=v3(0),x2=v3(1),x3=v3(2);
		
		Eigen::Vector3d eulerAng=euler_angles(eigVectors);
		Eigen::Vector3d kv;
		kv<<0,0,1;
		unghiuri<<eulerAng[1]<<" "<<acos(fabs(kv.dot(eigVectors.col(2))))<<endl;
		if(v[i]>0){
			output[i].push_back(i);
			output[i].push_back(v[i]);
			
			output[i].push_back(xcm[i]);
			output[i].push_back(ycm[i]);
			output[i].push_back(zcm[i]);
			
			output[i].push_back(axes[0]);
			output[i].push_back(axes[1]);
			output[i].push_back(axes[2]);
			
			output[i].push_back(x1);
			output[i].push_back(x2);
			output[i].push_back(x3);
			
			output[i].push_back(eulerAng[0]);
			output[i].push_back(eulerAng[1]);
			output[i].push_back(eulerAng[2]);
			
			
			
		}
		
		if(v[i]>0){
		for(int uu=0;uu<output[i].size();uu++){
			//cout<<output[i][uu]<<" ";
			analiza<<output[i][uu]<<" ";
		}
		//cout<<endl;
		analiza<<endl;
		}
	}
	
}

cubee getParticle(cubee& c,float culoare){
	
	int xmin=c.Nx;
	int xmax=0;
	int ymin=c.Ny;
	int ymax=0;
	int zmin=c.Nz;
	int zmax=0;
	
	for(int i1=0;i1<c.Nx;i1++)
	for(int i2=0;i2<c.Ny;i2++)
	for(int i3=0;i3<c.Nz;i3++)
	{
		if(c.get(i1,i2,i3)==culoare){
			if(i1>xmax)xmax=i1;
			if(i2>ymax)ymax=i2;
			if(i3>zmax)zmax=i3;
			if(i1<xmin)xmin=i1;
			if(i2<ymin)ymin=i2;
			if(i3<zmin)zmin=i3;
		}
	
	
	}
	
	cubee d(xmax+4-xmin,ymax+4-ymin,zmax+4-zmin);
	
	for(int i1=xmin;i1<=xmax;i1++)
	for(int i2=ymin;i2<=ymax;i2++)
	for(int i3=zmin;i3<=zmax;i3++){
		if(c.get(i1,i2,i3)==culoare){
			d.setSafe(i1-xmin+2,i2-ymin+2,i3-zmin+2,culoare);
		
		}
	}
	
	return d;

}


vector<double> volumeAnalysisOrientationColor(cubee& c,int color){
	vector<double> output;
	int v=0;
	float xcm;
	float ycm;
	float zcm;
	Eigen::Matrix3d m;
	
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		if(culoare==color){
		v+=1;
		xcm+=c.index3D(i)[0];
		ycm+=c.index3D(i)[1];
		zcm+=c.index3D(i)[2];
		}
	}

	if(v>0){
		xcm=xcm/v;
		ycm=ycm/v;
		zcm=zcm/v;
	}
	
	
	double x,y,z;
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		if(culoare==color){
		x=c.index3D(i)[0]-xcm;
		y=c.index3D(i)[1]-ycm;
		z=c.index3D(i)[2]-zcm;
		
		m(0,0)+=x*x;
		m(1,1)+=y*y;
		m(2,2)+=z*z;
		m(0,1)+=x*y;
		m(0,2)+=x*z;
		m(1,2)+=y*z;
		
	
		m(1,0)=m(0,1);
		m(2,0)=m(0,2);
		m(2,1)=m(1,2);	
		}
	}
	m=m/v;
	

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(m);
	Eigen::MatrixXd eigVectors=es.eigenvectors().real();
	Eigen::VectorXd eigValues=es.eigenvalues();
	
	Eigen::Vector3d axes;
	axes<<sqrt(5*eigValues(0)),sqrt(5*eigValues(1)),sqrt(5*eigValues(2));
	
	Eigen::Vector3d v1=eigVectors.col(0);
	Eigen::Vector3d v2=eigVectors.col(1);
	Eigen::Vector3d v3=eigVectors.col(2);
	
	Eigen::Vector3d eulerAng=euler_angles(eigVectors);

	output.push_back(color);
	output.push_back(v);
	output.push_back(xcm);
	output.push_back(ycm);
	output.push_back(zcm);
	output.push_back(axes[0]);
	output.push_back(axes[1]);
	output.push_back(axes[2]);
	output.push_back(eulerAng[0]);
	output.push_back(eulerAng[1]);
	output.push_back(eulerAng[2]);
	
	return output;
	
	
}


Eigen::Matrix3d momentOfInertia(cubee &cc,int culoare){
	Eigen::Matrix3d m;
	m<<0,0,0,0,0,0,0,0,0;
	double xcm=0,ycm=0,zcm=0;
	double vol=0;
	for(int i=0;i<cc.N_tot;i++){
		if(cc.get(i)==culoare){
			vol+=1;
			xcm+=cc.index3D(i)[0];
			ycm+=cc.index3D(i)[1];
			zcm+=cc.index3D(i)[2];
		}	
	}
	
	xcm/=vol*1.;
	ycm/=vol*1.;
	zcm/=vol*1.;
	
	double x,y,z;
	for(int i=0;i<cc.N_tot;i++){
		if(cc.get(i)==culoare){
			x=cc.index3D(i)[0]-xcm;
			y=cc.index3D(i)[1]-ycm;
			z=cc.index3D(i)[2]-zcm;
			
			
		
			m(0,0)+=x*x;//y*y+z*z;
			m(1,1)+=y*y;//x*x+z*z;
			m(2,2)+=z*z;//x*x+y*y;
			m(0,1)+=x*y;
			m(0,2)+=x*z;
			m(1,2)+=y*z;
			
		}
		m(1,0)=m(0,1);
		m(2,0)=m(0,2);
		m(2,1)=m(1,2);	
	}
	
	m/=vol*1.;
	return m;

}





vector<vector<double>> shapeAnalysisOrientationColor(cubee& c,int color){
	vector<double> output;
	int v=0;
	float xcm=0;
	float ycm=0;
	float zcm=0;
	Eigen::Matrix3d m;
	vector<int> puncteSuprafata;
	
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		if(culoare==color){
		v+=1;
		xcm+=c.index3D(i)[0];
		ycm+=c.index3D(i)[1];
		zcm+=c.index3D(i)[2];

		if(
			c.get(c.vecini(i,1,0,0))!=color||
			c.get(c.vecini(i,-1,0,0))!=color||
			c.get(c.vecini(i,0,1,0))!=color||
			c.get(c.vecini(i,0,-1,0))!=color||
			c.get(c.vecini(i,0,0,1))!=color||
			c.get(c.vecini(i,0,0,-1))!=color){
				puncteSuprafata.push_back(i);
				
			}
		}
	}
	
	if(v>0){
		xcm=xcm/v;
		ycm=ycm/v;
		zcm=zcm/v;
	}

	ofstream fis_raze("fis_raze");
	
	vector<vector<double>> puncte;
	for(int i=0;i<puncteSuprafata.size();i++){
			int poz=puncteSuprafata[i];
			
			double x=c.index3D(poz)[0]-xcm;
			double y=c.index3D(poz)[1]-ycm;
			double z=c.index3D(poz)[2]-zcm;
		
			double r=sqrt(x*x+y*y+z*z);
			double phi=atan2(y,x);
			double theta=acos(z/r);
			
					fis_raze<<r<<" "<<phi<<" "<<theta<<endl;
			
			vector<double> vvv={r,phi,theta};
			puncte.push_back(vvv);
	}
	
	return puncte;
	
	
	double x,y,z;
	for(int i=0;i<c.N_tot;i++){
		int culoare=c.get(i);
		if(culoare==color){
		x=c.index3D(i)[0]-xcm;
		y=c.index3D(i)[1]-ycm;
		z=c.index3D(i)[2]-zcm;
		
		m(0,0)+=x*x;
		m(1,1)+=y*y;
		m(2,2)+=z*z;
		m(0,1)+=x*y;
		m(0,2)+=x*z;
		m(1,2)+=y*z;
		
	
		m(1,0)=m(0,1);
		m(2,0)=m(0,2);
		m(2,1)=m(1,2);	
		}
	}
	m=m/v;
	

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(m);
	Eigen::MatrixXd eigVectors=es.eigenvectors().real();
	Eigen::VectorXd eigValues=es.eigenvalues();
	
	Eigen::Vector3d axes;
	axes<<sqrt(5*eigValues(0)),sqrt(5*eigValues(1)),sqrt(5*eigValues(2));
	
	Eigen::Vector3d v1=eigVectors.col(0);
	Eigen::Vector3d v2=eigVectors.col(1);
	Eigen::Vector3d v3=eigVectors.col(2);
	
	Eigen::Vector3d eulerAng=euler_angles(eigVectors);

	output.push_back(color);
	output.push_back(v);
	output.push_back(xcm);
	output.push_back(ycm);
	output.push_back(zcm);
	output.push_back(axes[0]);
	output.push_back(axes[1]);
	output.push_back(axes[2]);
	output.push_back(eulerAng[0]);
	output.push_back(eulerAng[1]);
	output.push_back(eulerAng[2]);
	
	//~ return output;
	
	
}








int main(){
	//testTest();
	return 0;


}
cubee readMRC(string fileName){
	ifstream fin(fileName.c_str(), std::ios::in | std::ios::binary);
    int32_t dim[3];
    fin.read(reinterpret_cast<char *>(dim), 3*sizeof(int32_t));
	cout<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<endl;
    cout<<"aici"<<endl;
	int n_elem = dim[0]*dim[1]*dim[2];
    cubee tomo(dim[0],dim[1],dim[2]);
	int32_t mode;
    fin.read(reinterpret_cast<char*>(&mode), sizeof(mode));

	
	int32_t nxstart;
    fin.read(reinterpret_cast<char*>(&nxstart), sizeof(int32_t));
	//cout<<nxstart<<endl;
	
	int32_t nystart;
    fin.read(reinterpret_cast<char*>(&nystart), sizeof(int32_t));
	//cout<<nxstart<<endl;
	
	int32_t nzstart;
    fin.read(reinterpret_cast<char*>(&nzstart), sizeof(int32_t));
	//cout<<nzstart<<endl;


	int32_t mx;
    fin.read(reinterpret_cast<char*>(&mx), sizeof(int32_t));
	//cout<<mx<<endl;
	
	int32_t my;
    fin.read(reinterpret_cast<char*>(&my), sizeof(int32_t));
	//cout<<my<<endl;
	
	int32_t mz;
    fin.read(reinterpret_cast<char*>(&mz), sizeof(int32_t));
	//cout<<mz<<endl;
	
	int32_t dim_cell[3];
    fin.read(reinterpret_cast<char *>(dim_cell), 3*sizeof(int32_t));
    //cout<<dim_cell[0]<<" "<<dim_cell[1]<<" "<<dim_cell[2]<<endl;
	//cout<<"aici"<<endl;
	fin.seekg(1024);
	
	int buf_len = sizeof(float) * n_elem;
	
	float *fvol = new float[buf_len];
	fin.read(reinterpret_cast<char*>(fvol), buf_len);
	fin.close();
	//~ #pragma omp simd 
	for(int i=0;i<n_elem;i++){
		tomo.Val[i]=fvol[i];
	}
	delete fvol;
	return tomo;

}
void writeMRC(cubee tomo,string fileName){
	
	
	ofstream fout(fileName.c_str(), std::ios::out | std::ios::binary);

   // header is 1024 bytes.  
   // 
   // first 12 bytes are 4byte x,y,z size values.
   int32_t dim[3];
   dim[0] = tomo.Nx; 
   dim[1] = tomo.Ny; 
   dim[2] = tomo.Nz;
    
   float* data; 
   // same values as 32-bit floats.
   
   float len[3];
   len[0] = (float)dim[0]*10; 
   len[1] = (float)dim[1]*10; 
   len[2] = (float)dim[2]*10;

   unsigned int buf_len = 3 * sizeof(int32_t);
   fout.write(reinterpret_cast<char *>(dim), 3 * sizeof(int32_t));

   // next 4 bytes determine the file type.
   int32_t mode = 2;
   fout.write( reinterpret_cast<char*>(&mode), sizeof(int32_t));

  // use value initialization to set all values to 0.
  // new char[N]() == calloc(N)
   char*   zeros = new char[1024]();
 
  // write out nxstart, nystart, nzstart.  all 0 is ignored. 3 float32 values.
   fout.write( reinterpret_cast<char*>(zeros), 12);
   
   // write mx,my,mz
   // same as nx, ny, nz
   fout.write(reinterpret_cast<char*>(dim), 3*sizeof(int32_t));
   
   // write cella (size of image in angstroms)
   // float values.
   fout.write(reinterpret_cast<char*>(len), 3*sizeof(float));
   
   // write cell angles in degrees (0 or 90?).
   fout.write(reinterpret_cast<char *>(zeros), 3*sizeof(int32_t));
   
   // mapping dimension of columns, rows, slices to axes.
   int32_t map_crs[3];
   map_crs[0] = 1; map_crs[1] = 2; map_crs[2] = 3;
   fout.write(reinterpret_cast<char*>(map_crs), 3*sizeof(int32_t));
   
   float stats[3];
   stats[0] = float(0);
   stats[1] = float(0);
   stats[2] = float(0);
   
   
   // min value in file/max value/avg value.
   fout.write(reinterpret_cast<char*>(stats), 3*sizeof(float));
   
   // write out a bunch of junk.  29 zero bytes.
   fout.write(reinterpret_cast<char *>(zeros), 30*sizeof(int32_t));
   
   // write the word MAP.
   fout.write("MAP ", 4);
   
   // everything else is zeros.
   fout.write(reinterpret_cast<char *>(zeros), 812);
   
   buf_len = tomo.N_tot * sizeof(float);
   data = new float[buf_len];
    
   // move from double to float.
   for(size_t i = 0; i < tomo.N_tot; i++)
   data[i] = tomo.Val[i];
   
	// write the data.
    fout.write(reinterpret_cast<char*>(data), buf_len);
  
    fout.close();
    delete[] data;
	delete[] zeros;
	
}

vector<float> MRCtoVector(string fileName){
	ifstream fin(fileName.c_str(), std::ios::in | std::ios::binary);
    int32_t dim[3];
    fin.read(reinterpret_cast<char *>(dim), 3*sizeof(int32_t));
	cout<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<endl;
    cout<<"aici"<<endl;
	int n_elem = dim[0]*dim[1]*dim[2];
    vector<float> Val(dim[0]*dim[1]*dim[2]);
	int32_t mode;
    fin.read(reinterpret_cast<char*>(&mode), sizeof(mode));

	
	int32_t nxstart;
    fin.read(reinterpret_cast<char*>(&nxstart), sizeof(int32_t));
	//cout<<nxstart<<endl;
	
	int32_t nystart;
    fin.read(reinterpret_cast<char*>(&nystart), sizeof(int32_t));
	//cout<<nxstart<<endl;
	
	int32_t nzstart;
    fin.read(reinterpret_cast<char*>(&nzstart), sizeof(int32_t));
	//cout<<nzstart<<endl;


	int32_t mx;
    fin.read(reinterpret_cast<char*>(&mx), sizeof(int32_t));
	//cout<<mx<<endl;
	
	int32_t my;
    fin.read(reinterpret_cast<char*>(&my), sizeof(int32_t));
	//cout<<my<<endl;
	
	int32_t mz;
    fin.read(reinterpret_cast<char*>(&mz), sizeof(int32_t));
	//cout<<mz<<endl;
	
	int32_t dim_cell[3];
    fin.read(reinterpret_cast<char *>(dim_cell), 3*sizeof(int32_t));
    //cout<<dim_cell[0]<<" "<<dim_cell[1]<<" "<<dim_cell[2]<<endl;
	//cout<<"aici"<<endl;
	fin.seekg(1024);
	
	int buf_len = sizeof(float) * n_elem;
	
	float *fvol = new float[buf_len];
	fin.read(reinterpret_cast<char*>(fvol), buf_len);
	fin.close();
	
	for(int i=0;i<n_elem;i++){
		Val[i]=fvol[i];
	}
	delete fvol;
	return Val;

}
