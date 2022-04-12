#import the module from the src file directory
from ../src import magn3t


#define a 3D empty space with dimesions 100x100x100
c=magn3t.cubee(100,100,100)

#add 3 spheres (x,y,z,radius)
magn3t.addSphere(c,40,40,50,18)
magn3t.addSphere(c,40,60,50,18)
magn3t.addSphere(c,60,40,50,18)

#write to the disk
magn3t.writeMRC(c,"test.mrc")

#apply threshold -> 0.5 is the limit , 10 is the color label for voxels which are above the limit
magn3t.applyThreshold_linear(c,0.5,10);

#distance map procedure (the label of the particles should be stated)
dist=magn3t.distanceMapGeneralEfficient(c,10);

#normalization should ensure a better control the parameters
dist.normalize()
magn3t.writeMRC(dist,"dist.mrc")

# a small quantity is substracted from the distance map
marker=dist-0.05;

#the local maxima are recovered
maxi=dist-magn3t.morphologicalReconstructionHybrid(marker,dist);
maxi.normalize()

magn3t.writeMRC(maxi,"maxi.mrc")


magn3t.applyThreshold_linear(maxi,0.01);
# magn3t.erode2(dist1,10,1);
# magn3t.dilate2(dist1,10,1);

# the maxima are filled with a value starting at 10 for labeling 
magn3t.fillParticlesRandom(maxi,10)

# the initial particles are filled separately starting from the maxima
flo=magn3t.priorityFlood(dist,maxi);
magn3t.writeMRC(flo,"filled.mrc");

# we analyse the labeled particles
magn3t.volumeAnalysis(flo) # a file "particleData" is written containting size information for each particle

