from mwispDBSCAN import MWISPDBSCAN
doDBSCAN = MWISPDBSCAN()

doDBSCAN.rawCOFITS="Q1Sub.fits"

doDBSCAN.averageRMS=0.5
#doDBSCAN.rmsFITS="yourRMS.fits"

doDBSCAN.pipeLine( getIntFITS=True, getCubeFITS=True  )

