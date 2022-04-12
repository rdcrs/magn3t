from src import magn3t as tomo


#define a 3D empty space with dimesions 100x100x100
c=tomo.cubee(100,100,100)

#add 3 spheres (x,y,z,radius)
tomo.addSphere(c,40,40,50,18)
tomo.addSphere(c,40,60,50,18)
tomo.addSphere(c,60,40,50,18)

#write to the disk
tomo.writeMRC(c,"test.mrc")

#apply threshold -> 0.5 is the limit , 10 is the color label for voxels which are above the limit
tomo.applyThreshold_linear(c,0.5,10);

#distance map procedure (the label of the particles should be stated)
dist=tomo.distanceMapGeneralEfficient(c,10);

#normalization should ensure a better control the parameters
dist.normalize()
tomo.writeMRC(dist,"dist.mrc")

# a small quantity is substracted from the distance map
marker=dist-0.05;

#the local maxima are recovered
maxi=dist-tomo.morphologicalReconstructionHybrid(marker,dist);
maxi.normalize()

tomo.writeMRC(maxi,"maxi.mrc")


tomo.applyThreshold_linear(maxi,0.01);
# tomo.erode2(dist1,10,1);
# tomo.dilate2(dist1,10,1);

# the maxima are filled with a value starting at 10 for labeling 
tomo.fillParticlesRandom(maxi,10)

# the initial particles are filled separately starting from the maxima
flo=tomo.priorityFlood(dist,maxi);
tomo.writeMRC(flo,"filled.mrc");

# we analyse the labeled particles
tomo.volumeAnalysis(flo) # a file "particleData" is written containting size information for each particle

