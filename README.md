# DBSCANpython3

example.py-----------------------------below

from mwispDBSCAN import MWISPDBSCAN  
doDBSCAN = MWISPDBSCAN()

doDBSCAN.rawCOFITS="Q1Sub.fits" #CO fits

doDBSCAN.averageRMS=0.5 #average RMS 
#doDBSCAN.rmsFITS="yourRMS.fits" #if you have a RMS for each spectra, use that instead

doDBSCAN.pipeLine( getIntFITS=True, getCubeFITS=True  ) #getIntFITS, produce the integrated intensity fits, getCubeFITS produce the crop data cube for each cloud, including the noise cube 

