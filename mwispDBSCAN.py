# This script is used reconstruct the script of DBSCAN.
# To make is more usefule to use. and can be imported septeratrely

import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.table import Table, vstack
# need to download myPYTHON.py
from myPYTHON import *
import os
from progressbar import *
import gc
import sys
from skimage.morphology import erosion, dilation
from scipy.ndimage import label, generate_binary_structure, binary_erosion, binary_dilation
from sklearn.cluster import DBSCAN
import requests

doFITS = myFITS()



# print doFITS.weighted_avg_and_std

class MWISPDBSCAN( object ):
    processPath = None  #"./"  # by default save the product in the current path
    backupPath =  "./DBSCANresults"
    rawCOFITS = None
    rmsFITS = None

    rawLabelFITSName = None
    rawCatName = None

    cleanCatName = None
    cleanLabelFITSName = None



    rmsData = None  # must has the same shape with Ny,and Nz of the data file

    averageRMS = None
    coreFITS = None

    ################# DBSCAN parameters

    cutoff_sigma = 2
    minPts = 4
    connectivity = 1

    ###################### seting of catalog selection
    saveCore = False
    minVox = 16
    minChannel = 3
    hasBeam = 1
    minPeakSigma = 5

    ###
    ##### parameters for cloud int

    cloudIntPath = None
    doFore=False #produce Foreround
    localVindex = None # either 0 or max(Nx), depends on Galacticl longitude
    doPureInt=False #produce pureInt

    #### running ID
    runID= None #only for cloud IT

    saveIntFITSname = None
    savePureIntFITSname = None
    saveMaskFITSname =  None
    saveForegroundFITSname = None
    cropExtend=2 #indeg

    regionName = None
    intFolderName = "cloudInt"

    areaThresh=0.015 #deg
    def __init__(self):
        # self.getModelTB()
        pass

    def setCloudIntParameters(self,cloudIntPath=None,doFore=False,doPureInt=False,localVindex=False):
        """

        :param cloudIntPath:
        :param doFore:
        :param doPureInt:
        :return:
        """
        self.cloudIntPath = cloudIntPath
        self.doFore = doFore
        self.doPureInt = doPureInt
        self.localVindex = localVindex



    def setDBSCANParameters(self, cutoff_sigma=2, minPts=4, connectivity=1):

        self.cutoff_sigma = cutoff_sigma
        self.minPts = minPts
        self.connectivity = connectivity

    def setCatalogSelectionCriteria(self, minVox=16, minChannel=3, hasBeam=1, minPeakSigma=5):
        self.minVox = minVox
        self.minChannel = minChannel
        self.hasBeam = hasBeam
        self.minPeakSigma = minPeakSigma

    def sumEdgeByCon1(self, extendMask):  # 7 in total

        # return  extendMask[1:-1, 1:-1, 1:-1] + extendMask[0:-2, 1:-1, 1:-1] +  extendMask[2:, 1:-1, 1:-1] + extendMask[1:-1, 0: -2, 1:-1]+ extendMask[1:-1, 2:, 1:-1]+ extendMask[1:-1, 1:-1, 0:-2] +   extendMask[1:-1, 1:-1, 2:]
        raw = extendMask[1:-1, 1:-1, 1:-1]

        leftShiftZ = extendMask[0:-2, 1:-1, 1:-1]
        rightShiftZ = extendMask[2:, 1:-1, 1:-1]

        leftShiftY = extendMask[1:-1, 0: -2, 1:-1]
        rightShiftY = extendMask[1:-1, 2:, 1:-1]

        leftShiftX = extendMask[1:-1, 1:-1, 0:-2]
        rightShiftX = extendMask[1:-1, 1:-1, 2:]

        sumAll = raw + leftShiftZ + rightShiftZ + leftShiftY + rightShiftY + leftShiftX + rightShiftX

        return sumAll

    def sumEdgeByCon2(self, extendMask):  # 27 in total
        sumAll = extendMask[1:-1, 1:-1, 1:-1] * 0
        Nz, Ny, Nx = sumAll.shape
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:

                    if np.sqrt(abs(i) + abs(j) + abs(k)) > 1.5:
                        continue

                    sumAll = sumAll + extendMask[1 + i:Nz + 1 + i, j + 1:Ny + 1 + j, k + 1: Nx + 1 + k]

        return sumAll

    def sumEdgeByCon3(self, extendMask):  # 27 in total
        raw = extendMask[1:-1, 1:-1, 1:-1]
        Nz, Ny, Nx = raw.shape
        sumAll = raw * 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    sumAll = sumAll + extendMask[1 + i:Nz + 1 + i, j + 1:Ny + 1 + j, k + 1: Nx + 1 + k]

        return sumAll

    ############

    def setOutputPath(self, targetPath):

        if os.path.isdir(targetPath):
            self.processPath = targetPath
        else:
            try:
                os.mkdir(targetPath)
                self.processPath = targetPath
            except:
                print("The target output path does not exist, please check your path: ", targetPath)
                sys.exit()

        self.getOutputNames()

    def getOutputNames(self):
        """
        create all the outpute names of files, even they have not been created
        :return:
        """

        if self.rawCOFITS is None:
            print("CO FITS is necessary")
            return
        baseName = os.path.basename(self.rawCOFITS)
        saveTag = self.getDBSCANTag()



        preFix = os.path.join(self.processPath, baseName[0:-5] + saveTag)

        self.rawLabelFITSName = preFix + ".fits"
        self.cleanLabelFITSName = preFix + "_Clean.fits"

        self.rawCatName = preFix + ".fit"
        self.cleanCatName = preFix + "_Clean.fit"

        # if self.rmsFITS is None: #no rms fits is provided

    def getDBSCANTag(self):

        """
        return
        :param cutoff: #the sigma to which the overall datacube will be cut
        :param minpts:
        :param conType: #thre are
        :return:
        """

        saveName = "dbscanS{}P{}Con{}".format(self.cutoff_sigma, self.minPts, self.connectivity)

        return saveName

    def getUniformRMSData(self, averageRMS, Ny, Nx):

        return averageRMS + np.zeros((Ny, Nx))

    def getCoreArray(self, extendMask):
        """

        :param maskArray:
        :param contype:
        :return:
        """

        if self.connectivity == 1:
            coreArray = self.sumEdgeByCon1(extendMask)

        if self.connectivity == 2:
            coreArray = self.sumEdgeByCon2(extendMask)

        if self.connectivity == 3:
            coreArray = self.sumEdgeByCon3(extendMask)
        return coreArray

    def computeDBSCAN(self, splitN=1):
        """
        There are two steps of computing DBSCAN, compute the DBSCAN label, 2 extract catlog, select post catalog and produce clean fits, would be doe seperatrely
        :param COdata:
        :param COHead:
        :param min_sigma: cutoff
        :param min_pix:
        :param connectivity:
        :param region:
        :param getMask:
        :param savePath:
        :param mimicDendro:
        :param rmsFITS:
        :param inputRMS:
        :return:
        """
        # pass
        if self.processPath is None:
            if not os.path.isdir(self.backupPath):
                os.mkdir(self.backupPath )
            self.processPath=self.backupPath


        self.getOutputNames()

        if self.rawCOFITS is None:
            print("rawCOFITS need to be provided, stoping...")
            return


        COdata, COHead = doFITS.readFITS(self.rawCOFITS)

        if len(COdata.shape) == 4:
            COdata = COdata[0]

        Nz, Ny, Nx = COdata.shape

        if self.averageRMS is None and self.rmsFITS is None:
            print(
                "You need either use averaged rms, or use rmsFITS, which can be accurate to each spectrum. Please check you input")

        if self.rmsFITS == None:  # use average rms, te rmsFITS has a higher priority

            rmsData = self.getUniformRMSData(self.averageRMS, Ny, Nx)

        else:
            print("Using rms fits...", self.rmsFITS)
            rmsData, rmsHead = myFITS.readFITS(self.rmsFITS)
        self.rmsData = rmsData  # record the rms Data

        rmsCOData = COdata / rmsData

        del COdata
        gc.collect()

        goodValues = rmsCOData >= self.cutoff_sigma

        del rmsCOData
        gc.collect()

        s = generate_binary_structure(3, self.connectivity)

        overlapping = 10

        indexEdges = np.linspace(0, Nx - 1, splitN + 1, dtype=np.int32)
        coreArray = np.zeros_like(goodValues, dtype=np.int32)

        for i in range(len(indexEdges) - 1):
            ####
            print("doing", i)
            indexRange = [indexEdges[i] - overlapping, indexEdges[i + 1] + overlapping + 1]

            leftCut = overlapping
            rightCut = indexEdges[i + 1] - indexEdges[i] + 1 + overlapping

            if i == 0:
                indexRange[0] = 0
                leftCut = 0
                rightCut = indexEdges[i + 1] - indexEdges[i] + 1

            if i == len(indexEdges) - 1:
                indexRange[1] = indexEdges[i + 1] + 1
                rightCut = indexEdges[i + 1] - indexEdges[i] + 1 + overlapping

            subGood = goodValues[:, :, indexRange[0]: indexRange[1]]




            subNz, subNy, subNx = subGood.shape
            subExtend = np.zeros([subNz + 2, subNy + 2, subNx + 2], dtype=np.int32)
            subExtend[1:-1, 1:-1, 1:-1] = subGood


            subCoreArray = self.getCoreArray(subExtend)


            coreArray[:, :, indexEdges[i]:  indexEdges[i + 1] + 1] = subCoreArray[:, :, leftCut: rightCut]
            del subCoreArray
            del subExtend
            gc.collect()
        # coreArray = self.getCoreArray(extendMask)



        coreArray = coreArray >= self.minPts
        gc.collect()

        coreArray[
            ~goodValues] = False  # nan could be, #remove falsely, there is a possibility that, a bad value may have lots of pixels around and clould be
        # coreArray = coreArray + 0


        labeled_core, num_features = label(coreArray,
                                           structure=s,output=np.int32 )  # first label core, then expand, otherwise, the expanding would wrongly connected

        print(num_features, "features found!")


        del coreArray
        gc.collect()
        if self.saveCore:
            """
            """
            saveCoreName = self.getLabelCoreFITSName()
            fits.writeto(saveCoreName, labeled_core, header=COHead, overwrite=True)
            self.coreFITS = saveCoreName

        selectExpand = np.logical_and(labeled_core == 0, goodValues)
        del goodValues
        gc.collect()
        # expand labeled_core
        # coreLabelCopy=labeled_core.copy()

        expandTry = dilation(labeled_core,
                             s)  # first try to expand, then only keep those region that are not occupied previously
        # it is possible  that a molecular cloud may have less pixN than 8, because of the closeness of two
        # only expanded, one time
        labeled_core[selectExpand] = expandTry[selectExpand]

        fits.writeto(self.rawLabelFITSName, labeled_core, header=COHead, overwrite=True)

        del labeled_core
        gc.collect()

    def getLabelCoreFITSName(self):
        """
        used to get the DBSCAN label fits name, when you do not want to rerun the DBSCAN process
        :return:
        """
        baseName = os.path.basename(self.rawCOFITS)
        saveTag = self.getDBSCANTag()

        saveLabelFITSNameCore = os.path.join(self.processPath, baseName[0:-5] + saveTag + "_core.fits")

        return saveLabelFITSNameCore

    def getIndices(self, Z0, Y0, X0, values1D, choseID):

        cloudIndices = np.where(values1D == choseID)

        cX0 = X0[cloudIndices]
        cY0 = Y0[cloudIndices]
        cZ0 = Z0[cloudIndices]

        return tuple([cZ0, cY0, cX0])

    def rmsmap(self, outPUT=None, overwrite=True):
        """
        3d rms
        :param outPUT:
        :param overwrite:
        :return:
        """

        if self.rawCOFITS is None:
            print("rawCOFITS need to be provided, stopping...")
            return

        COdata, COHead = doFITS.readFITS(self.rawCOFITS)

        if len(COdata.shape) == 4:
            COdata = COdata[0]

        if outPUT is None:
            writeName = "rmsmap.fits"

        else:
            writeName = outPUT

        fileExist = os.path.isfile(writeName)

        if overwrite and fileExist:
            os.remove(writeName)

        Nz, Ny, Nx = COdata.shape

        rmsData = np.zeros_like(COdata, dtype=np.float32)

        for i in range(Nz):
            channelI = COdata[i]
            negativeValues = channelI[channelI < 0]

            sigma = np.std(negativeValues) / np.sqrt(1 - 2. / np.pi)

            print(sigma)
            rmsData[i, :, :] = sigma

        fits.writeto(writeName, rmsData, header=COHead)
        return fits.open(writeName)[0]

    def getEmptyCat(self, getEmptyRow=False):
        """

        :return:
        """

        #############
        newTB = Table(names=("_idx", "area_exact", "v_cen", "v_rms", \
                             "x_cen", "y_cen", "sum", "l_rms", \
                             "b_rms", "pixN", "peak", "peakL", \
                             "peakB", "peakV", "area_accurate", "lineWidth", "allChannel", "has22", "touchL",  "touchB",  "touchLB",  "touchV",
                             "complete","meanSNR"), dtype=('i8', "f8", "f8", "f8", \
                                                 "f8", "f8", "f8", "f8", \
                                                 "f8", "i8", "f8", "i8", \
                                                 "i8", "i8", "f8", "f8", "i8", "i4", "i4", "i4", "i4","i4","i4","f8")

                      )
        newTB["area_accurate"].unit = "arcmin2"

        newTB["area_exact"].unit = "arcmin2"
        newTB["v_cen"].unit = "km/s"
        newTB["v_rms"].unit = "km/s"

        newTB["x_cen"].unit = "deg"
        newTB["y_cen"].unit = "deg"
        newTB["l_rms"].unit = "deg"
        newTB["b_rms"].unit = "deg"

        newTB["peakB"].unit = ""
        newTB["peakL"].unit = ""
        newTB["peakV"].unit = ""
        newTB["peak"].unit = "K"
        newTB["sum"].unit = "K"

        newTB["allChannel"].unit = ""
        newTB["has22"].unit = ""
        newTB["touchLB"].unit = ""
        newTB["touchL"].unit = ""
        newTB["touchB"].unit = ""
        newTB["touchV"].unit = ""
        newTB["complete"].unit = ""
        newTB["meanSNR"].unit = ""
        if getEmptyRow:
            newTB.add_row()
            return newTB[0]

        return newTB



    def getDBSCANCatlog(self, doClean=True):
        """
        Extract catalog from label fits, the minPix and rms is only used for saving
        #need a TB Model, if TB model does not exist, better downlooad from online
        This catalog has already been cleaned
        four criteria minPix"
        if you want to change the catlog selection criteria, please see setCatalogSelectionCariteria
        :param labelArray:
        :param head:
        :return:
        """
        #### Add a column of line width, which will be used in statistics
        ####

        self.getOutputNames()

        if self.rawLabelFITSName is None or not os.path.isfile(self.rawLabelFITSName):
            print("The DBSCAN label FITS is needed  ")
            return

        dataCluster, headCluster = myFITS.readFITS(self.rawLabelFITSName)

        # calculate the area of one pixel

        sizeL = headCluster["CDELT1"] * 60
        sizeB = headCluster["CDELT2"] * 60

        sizeB = abs(sizeB)  # in arcmin
        sizeL = abs(sizeL)  # in arcmin

        pixelArea = sizeB * sizeL

        Nz, Ny, Nx = dataCluster.shape

        if self.rawCOFITS is None:
            print("A COFITS is needed  ")
            return

        dataCO, headCO = myFITS.readFITS(self.rawCOFITS)

        if len(dataCO.shape) == 4:
            dataCO = dataCO[0]

        if self.rmsFITS is not None:
            print("Using rms fits...", self.rmsFITS)
            rmsData, rmsHead = myFITS.readFITS(self.rmsFITS)
            self.rmsData = rmsData

        else:

            if self.averageRMS is not None and self.rawCOFITS is not None:
                rmsData = self.getUniformRMSData(self.averageRMS, Ny, Nx)

                self.rmsData = rmsData

        if self.rmsData is None:
            print("A rmsData is needed to select peakSgima")
            return

        minV = np.nanmin(dataCluster[0])  # noise mask value

        wcsCloud = WCS(headCluster, naxis=3)

        wcsCloud.wcs.bounds_check(False, False)  # for molecular clouds toward the anti Galacic center

        clusterIndex1D = np.where(dataCluster > minV)
        clusterValue1D = dataCluster[clusterIndex1D]

        del dataCluster
        gc.collect()
        dataZero = np.zeros_like(dataCO)  #

        Z0, Y0, X0 = clusterIndex1D

        newTB = self.getEmptyCat(getEmptyRow=False)

        zeroProjection = np.zeros((Ny, Nx))  # one zero channel, used to get the projection area and
        zeroProjectionExtend = np.zeros((Ny + 1, Nx + 1))

        idCol = "_idx"

        # count all clusters

        # ids,count=np.unique(dataCluster,return_counts=True )
        ids, count = np.unique(clusterValue1D, return_counts=True)
        GoodIDs = ids
        GoodCount = count
        print("Total number of trunks? ", len(ids))
        # print "Total number of Good Trunks? ",len(GoodIDs)

        # dataCO,headCO=doFITS.readFITS( CO12FITS )
        widgets = ['Calculating cloud parameters: ', Percentage(), ' ', Bar(marker='0', left='[', right=']'), ' ',
                   ETA(),
                   ' ', FileTransferSpeed()]  # see docs for other options

        catTB = newTB

        # zeroP
        # remove any cloud with voxels less than minPix
        selectVox = GoodCount >= self.minVox  # #criteria 1

        if doClean:
            GoodCount = GoodCount[selectVox]
            GoodIDs = GoodIDs[selectVox]
            print(len(GoodIDs), "cluster has more than {} voxels.".format(self.minVox))

        pbar = ProgressBar(widgets=widgets, maxval=len(GoodIDs))
        pbar.start()

        for i in range(len(GoodIDs)):

            # i would be the newID
            newID = GoodIDs[i]

            pixN = GoodCount[i]

            newRow = self.getEmptyCat(getEmptyRow=True)

            newRow[idCol] = newID

            cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, newID)
            cloudV = cloudIndex[0]
            cloudB = cloudIndex[1]
            cloudL = cloudIndex[2]

            coValues = dataCO[cloudIndex]
            rmsValues = self.rmsData[(cloudB,cloudL)]

            # step1, check peak
            # peak should be done first
            # add three columns, to recrod if the cloud touch the edge of fits
            if np.min(cloudV) == 0 or np.max(cloudV) == Nz - 1:
                newRow["touchV"] = 1

            if np.min(cloudB) == 0 or np.max(cloudB) == Ny - 1:
                newRow["touchLB"] = 1
                newRow["touchB"] = 1

            if np.min(cloudL) == 0 or np.max(cloudL) == Nx - 1:
                newRow["touchLB"] = 1
                newRow["touchL"] = 1

            if not newRow["touchLB"] and not newRow["touchV"]:
                newRow["complete"] = 1

            peakIndex = coValues.argmax()
            peakV = cloudV[peakIndex]
            peakB = cloudB[peakIndex]
            peakL = cloudL[peakIndex]
            peak = coValues[peakIndex]
            # #criteria 2
            meanSNR= np.mean(coValues/rmsValues)
            peakSimga = 0

            if len(self.rmsData.shape) == 2:
                peakSimga = self.minPeakSigma * self.rmsData[peakB, peakL]
            if len(self.rmsData.shape) == 3:
                peakSimga = self.minPeakSigma * self.rmsData[peakV, peakB, peakL]

            if peak < peakSimga and doClean:  # accurate to lines
                continue  # do not consider the minimum peaks

            # #criteria 3, check consecutive channels

            ############### ###
            diffVs = np.unique(cloudV)

            if len(diffVs) < self.minChannel and doClean:  # reject all cloud that has channels less than 3 channels
                continue

            # get the exact peak position, which would be used to

            projectIndex = tuple([cloudB, cloudL])

            zeroProjection[projectIndex] = 1
            zeroProjectionExtend[0:-1, 0:-1] = zeroProjection
            sum22 = zeroProjectionExtend[0:-1, 0:-1] + zeroProjectionExtend[0:-1, 1:] + zeroProjectionExtend[1:,
                                                                                        0:-1] + zeroProjectionExtend[1:,
                                                                                                1:]
            projectionPixN = np.sum(zeroProjection)
            # if any pixel>4:
            #

            if 4 in sum22:
                newRow["has22"] = 1

            else:  # Serious bug
                newRow["has22"] = 0

            if self.hasBeam and newRow["has22"] == 0 and doClean:  # #criteria 4
                zeroProjection[projectIndex] = 0
                zeroProjectionExtend[0:-1, 0:-1] = zeroProjection
                continue
            # newRow["has22"] = 0

            # calculate the accurate, this is only used for data at high Galactic latitude, and useless form the MWISP survey
            # to increase the speed, this part is commentted

            # indexB2D,indexL2D=np.where(zeroProjection==1 )
            # _,BS2D, LS2D = wcsCloud.wcs_pix2world(indexL2D,indexB2D,  0, 0)
            # area_accurate=np.sum( np.cos( np.deg2rad(BS2D) )    )*0.25
            # newRow["area_accurate"]= area_accurate

            sumCO = np.sum(coValues, dtype=np.float64)  # float32 is not enough for large molecular clouds

            Vcen, Vrms = doFITS.weighted_avg_and_std(cloudV, coValues)
            Bcen, Brms = doFITS.weighted_avg_and_std(cloudB, coValues)
            Lcen, Lrms = doFITS.weighted_avg_and_std(cloudL, coValues)

            # calculate the exact area

            # LBcore = zip(cloudB, cloudL)
            # pixelsN= {}.fromkeys(LBcore).keys() #len( set(LBcore) )
            # area_exact=len(pixelsN)*0.25 #arc mins square
            area_exact = np.sum(zeroProjection) * pixelArea

            newRow["area_exact"] = area_exact

            # dataClusterNew[cloudIndex] =newID

            # save values
            newRow["pixN"] = pixN
            newRow["peak"] = peak

            newRow["peakV"] = peakV
            newRow["peakB"] = peakB
            newRow["peakL"] = peakL
            newRow["meanSNR"] = meanSNR
            # newRow["peak2"]= peak2

            newRow["sum"] = sumCO

            newRow["x_cen"], newRow["y_cen"], newRow["v_cen"] = wcsCloud.wcs_pix2world(Lcen, Bcen, Vcen, 0)
            newRow["v_cen"] = newRow["v_cen"] / 1000.
            dv = headCluster["CDELT3"] / 1000.  # km/s

            dl = abs(headCluster["CDELT1"])  # deg

            newRow["v_rms"] = Vrms * dv
            newRow["l_rms"] = Lrms * dl
            newRow["b_rms"] = Brms * dl

            # _, Nchan=np.unique( cloudV, return_counts=True)

            # newRow["Nchannel"] =    np.max(P3Value)# if there is a three consecutive spectra in the cloud
            newRow["allChannel"] = len(diffVs)

            zeroProjection[projectIndex] = 0
            zeroProjectionExtend[0:-1, 0:-1] = zeroProjection

            ##################################################below are equivalenet legnth
            dataZero[cloudIndex] = dataCO[cloudIndex]

            # cropThe cloudRange
            minY = np.min(cloudB)
            maxY = np.max(cloudB)
            ###########
            minX = np.min(cloudL)
            maxX = np.max(cloudL)

            ###########
            minZ = np.min(cloudV)
            maxZ = np.max(cloudV)

            #########

            cloudCropSpectra = dataZero[:, minY:maxY + 1, minX:maxX + 1]
            averageSpectraCrop = np.nansum(cloudCropSpectra, axis=(1, 2))

            # count the number spectra

            totalSpectral = projectionPixN / 1.

            meanSpectral = averageSpectraCrop / 1. / totalSpectral

            # if 0:
            # savefileName = saveSpectralPath + "{}_{}Spectral".format(regionName, testID)

            # np.save(savefileName, [vAxis, meanSpectral])

            spectraPeak = np.max(meanSpectral)

            area = dv * np.sum(meanSpectral)

            eqLineWidth = area / spectraPeak
            dataZero[cloudIndex] = 0

            newRow["lineWidth"] = eqLineWidth

            catTB.add_row(newRow)

            pbar.update(i)

        pbar.finish()
        # save the clouds

        # fits.writeto(self.regionName+"NewCloud.fits", dataClusterNew,header=headCluster,overwrite=True   )

        if doClean:
            catTB.write(self.cleanCatName, overwrite=True)

        else:
            catTB.write(self.rawCatName, overwrite=True)

    def cleanTB(self):

        """
        select clouds that satifies the selection criteria
        :return:
        """

        if self.rawCatName is None:
            print("no catalog file provided")

            return

        if self.rmsData is None:
            print("You need to provide the rms FITS data to select peak accordingly")

        rawTB = Table.read(self.rawCatName)

        # first voxel
        filterTB = rawTB[rawTB["pixN"] >= self.minVox]

        filterTB = filterTB[filterTB["has22"] == self.hasBeam]

        # select by minCkannel
        filterTB = filterTB[filterTB["allChannel"] >= self.minChannel]

        # second peak

        peakB = list(map(int, filterTB["peakB"]))
        peakL = list(map(int, filterTB["peakL"]))
        peakCoordinate = tuple([peakB, peakL])

        rmsArray = self.rmsData[peakCoordinate]
        filterTB = filterTB[filterTB["peak"] >= self.minPeakSigma * rmsArray]

        filterTB.write(self.cleanCatName, overwrite=True)

    def getCloudIntName(self,cloudID):
        """

        :return:
        """
        self.runID  = cloudID

        self.saveIntFITSname =  os.path.join(self.cloudIntPath, "Cloud{}_int.fits".format(self.runID))
        self.savePureIntFITSname =  os.path.join(self.cloudIntPath , "Cloud{}_PureInt.fits".format(self.runID))
        self.saveMaskFITSname =  os.path.join(self.cloudIntPath , "Cloud{}_mask.fits".format( self.runID ))
        self.saveForegroundFITSname = os.path.join(self.cloudIntPath , "Cloud{}_fore.fits".format( self.runID ))

    def getCloudIntPath(self  ):
        self.cloudIntPath =  os.path.join(self.processPath, self.intFolderName )

        if not os.path.isdir( self.cloudIntPath ):
            os.mkdir( self.cloudIntPath )


    def produceCloudIntFITS(self,  inputTB=None, outputPath=None, pureInt=False, foreground=False, localVindx=None, onID=None,  doCrop=False,cropExtend=None):
        """

        :param inputTB: in case, a sub set of the DBSCAN clean TB is intested, other would be ignore
        :param outputPath: outPath of int and mask FITS
        :param pureInt: produce a int map, with noise voxels masked
        :param foreground: local V is need to produce a foreground emission map based on the velocity, used for distance examination
        :param localV: #in km/s, used to produce foreground fits
        :param onID: # in case only one cloud is interested
        :param areaThresh: The minimum area for distance examination
        :param cropExtend:  crop The int FITS, incase the wholse area is large

        :return:
        """


        if self.cleanCatName is None or not os.path.isfile( self.cleanCatName )  :
            print("Please get the DBSCAN data ready 1")
            return

        if  self.rawCOFITS is None or not os.path.isfile( self.rawCOFITS ) :
            print("Please get the DBSCAN data ready 2")
            return

        if  self.cleanLabelFITSName  is None or not os.path.isfile( self.cleanLabelFITSName ) :
            print("Please get the DBSCAN data ready 3")
            return

        if self.processPath is None:
            print("Please get the DBSCAN data ready 4")
            return

        if cropExtend is not None:
            self.cropExtend=cropExtend

        if outputPath is None  :
            self.getCloudIntPath()

        else:
            self.cloudIntPath = outputPath
        print("Saving cloud int FITS to ",self.cloudIntPath)
        if localVindx is not None:
            self.localVindex = localVindx
        #####
        TB=Table.read(self.cleanCatName)

        if inputTB is not None  :
            TB = inputTB


        TB.sort("area_exact")
        TB.reverse()

        TB = TB[TB["area_exact"] >= self.areaThresh * 3600]  ## Remove small molecular clouds

        if onID is not None: #if only cares about one molecular cloud
            TB=TB[ TB["_idx"] == onID  ]


        dataCluster, headCluster = myFITS.readFITS(self.cleanLabelFITSName)


        clusterIndex1D = np.where(dataCluster > 0)
        clusterValue1D = dataCluster[clusterIndex1D]
        Z0, Y0, X0 = clusterIndex1D
        del dataCluster # release the memory
        gc.collect()


        dataCO, headCO = myFITS.readFITS(self.rawCOFITS )
        coSpec0, vaxis0 = doFITS.getSpectraByIndex(dataCO, headCO, 0, 0)

        #### used to produce precise foreground emission
        dataCOMask =     np.zeros_like(dataCO, dtype=np.float32)
        dataCOMask[ clusterIndex1D ] =  dataCO[clusterIndex1D]
        ####################




        velsolution = vaxis0[1] - vaxis0[0]

        ################
        projection0 = np.zeros_like(dataCO[0] )

        widgets = ['Integrating cloud fits:', Percentage(), ' ', Bar(marker='0', left='[', right=']'), ' ', ETA(), ' ',
                   FileTransferSpeed()]  # see docs for other options

        pbar = ProgressBar(widgets=widgets, maxval=len(TB))
        pbar.start()
        indexRun = 0

        zeroCube = np.zeros_like(dataCO, dtype=np.float32)

        for eachDBRow in TB:
            indexRun = indexRun + 1
            pbar.update(indexRun)
            cloudID = eachDBRow["_idx"]

            self.getCloudIntName(cloudID) ####

            cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, cloudID)
            iz, iy, ix = cloudIndex

            projectIndices = tuple([iy, ix])
            ############ determine v range
            centerV = eachDBRow["v_cen"]
            dv = eachDBRow["v_rms"]
            vRange = [centerV - dv * 3, centerV + dv * 3]
            startV0, endVindex = self.getVindexRange(vaxis0, vRange)

            cropCOCube = dataCO[startV0:endVindex + 1]


            ###int fits
            sumCO = np.sum(cropCOCube, axis=0, dtype=float) * velsolution

            fits.writeto(self.saveIntFITSname , sumCO, header=headCluster, overwrite=True)

            ##mask fits
            projection0[projectIndices] = 1

            fits.writeto(self.saveMaskFITSname, projection0, header=headCluster, overwrite=True)

            ##foreground fits
            if pureInt:  # usually do no use this
                #startV0 = min(iz)
                #endV0 = max(iz)

                zeroCube[cloudIndex] = dataCO[cloudIndex]
                cropCOCubePure = zeroCube[startV0:endVindex + 1]

                sumCOPure = np.sum(cropCOCubePure, axis=0, dtype=float) * velsolution
                fits.writeto(savePureIntFITSname, sumCOPure, header=headCluster, overwrite=True)

            if foreground:  # the way of generating foreground fits is different, the following code is used to produce foreground fits for Q2
                #### using the close one

                if abs(endVindex-self.localVindex) < abs(startV0-self.localVindex):

                    starForeI = np.min( [endVindex, self.localVindex   ] )
                    endForeI = np.max( [endVindex, self.localVindex   ] )
                    foreCOCube = dataCOMask[starForeI  : endForeI ]

                else:
                    starForeI = np.min( [startV0, self.localVindex   ] )
                    endForeI = np.max( [startV0, self.localVindex   ] )
                    foreCOCube = dataCOMask[starForeI  : endForeI ]


                sumForeground = np.sum(foreCOCube, axis=0, dtype=float) * velsolution

                fits.writeto(self.saveForegroundFITSname , sumForeground, header=headCluster, overwrite=True)

            if doCrop: #crop the fits
                cropLrange, cropBrange = doFITS.getLBrangeWithMask(self.saveMaskFITSname,extendByDeg= self.cropExtend )

                doFITS.cropFITS2D(self.saveIntFITSname,outFITS=self.saveIntFITSname,Lrange=cropLrange,Brange=cropBrange,  overWrite=True)
                doFITS.cropFITS2D(self.savePureIntFITSname,outFITS=self.savePureIntFITSname,Lrange=cropLrange,Brange=cropBrange,  overWrite=True)
                doFITS.cropFITS2D(self.saveForegroundFITSname,outFITS=self.saveForegroundFITSname,Lrange=cropLrange,Brange=cropBrange,  overWrite=True)
                doFITS.cropFITS2D(self.saveMaskFITSname,outFITS=self.saveMaskFITSname,Lrange=cropLrange,Brange=cropBrange,  overWrite=True)


            projection0[projectIndices] = 0
            zeroCube[cloudIndex] = 0

            # zeroCluster[cloudIndex] = cloudID  # remove this cluster and do not record this cluster

        pbar.finish()

    def getVindexRange(self, vaxis, vRange):
        """

        :param vaxis:
        :param vRange:
        :return:
        """

        minValue = min(vRange)
        maxValue = max(vRange)

        indexV0 = doFITS.find_nearestIndex(vaxis, minValue)
        indexV1 = doFITS.find_nearestIndex(vaxis, maxValue)

        return [indexV0, indexV1]

    def produceCleanFITS(self):
        """
        remove noise cluster according to the label fits and catFITS name
        :return:
        """
        if self.rawLabelFITSName is None:
            print("You need to provde the label FITS  ")

            return

        if self.cleanCatName is not None and os.path.isfile(self.cleanCatName):
            targetTBFile = self.cleanCatName  # table has alread been cleaned

        if not os.path.isfile(self.cleanCatName) and not os.path.isfile(self.rawCatName):
            print("You need to provde a raw or a cleaned fits first to produce a clean fits")
            return

        if os.path.isfile(self.rawCatName) and not os.path.isfile(self.cleanCatName):
            # catalog has been calculated but not cleaned, need to clean table first, then do clean
            self.cleanTB()
            targetTBFile = self.cleanCatName  # table has alread been cleaned

        TB = Table.read(targetTBFile)
        dataCluster, headCluster = doFITS.readFITS(self.rawLabelFITSName)
        noiseLabel = np.min(dataCluster[0])
        clusterIndex1D = np.where(dataCluster > noiseLabel)
        clusterValue1D = dataCluster[clusterIndex1D]
        Z0, Y0, X0 = clusterIndex1D

        #

        widgets = ['Cleaning label fits:', Percentage(), ' ', Bar(marker='0', left='[', right=']'), ' ', ETA(), ' ',
                   FileTransferSpeed()]  # see docs for other options

        pbar = ProgressBar(widgets=widgets, maxval=len(TB))
        pbar.start()

        indexRun = 0

        returnCluster = np.zeros_like(dataCluster) + noiseLabel

        for eachDBRow in TB:
            indexRun = indexRun + 1
            pbar.update(indexRun)
            cloudID = eachDBRow["_idx"]

            cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, cloudID)
            returnCluster[cloudIndex] = cloudID  # remove this cluster and do not record this cluster

        pbar.finish()
        fits.writeto(self.cleanLabelFITSName, returnCluster, header=headCluster, overwrite=True)

    def produceMask(self, COFITS, labelFITS, outFITS=None):
        """
        produce a masked COFITS, with labelsFITS
        :param COFITS:
        :param labelFITS:
        :return:
        """

        dataCO, headCO = doFITS.readFITS(COFITS)

        dataLabel, headLabel = doFITS.readFITS(labelFITS)

        noiseLabel = np.min(dataLabel[0])

        if outFITS is None:
            outFITS = "mask_" + COFITS

        # maskCO
        dataCO[dataLabel == noiseLabel] = np.nan

        fits.writeto(outFITS, dataCO, header=headCO, overwrite=True)

    def produceIndividualClouds(self,  savePath=None, noiseMask=0,
                                inputTB=None):
        """


        Revisement: By qzyan for adding a noise cube for each individual clouds
        #output all data cubes for each cloud

        :return:
        """

        #################

        # savePath=""
        self.getOutputNames()

        if savePath is None:
            savePath = os.path.join(self.processPath,"cloudSubCubes")

        if os.path.isdir(savePath):
            pass
        else:
            os.makedirs(savePath)

        #####
        TB=Table.read(self.cleanCatName)

        if inputTB is not None  :
            TB = inputTB


        TB.sort("area_exact")
        TB.reverse()

        cloudTB = TB[TB["area_exact"] >= self.areaThresh * 3600]  ## Remove small molecular clouds


        # cloudTB=self.removeWrongEdges(cloudTB)
        print(len(cloudTB), " molecular clouds in total.")



        dataCluster, headCluster = myFITS.readFITS(self.cleanLabelFITSName)
        dataCO, headCO = myFITS.readFITS(self.rawCOFITS)
        # print cloudTB

        minV = np.nanmin(dataCluster[0])
        wcsCloud = WCS(headCluster, naxis=3)
        clusterIndex1D = np.where(dataCluster > minV)
        clusterValue1D = dataCluster[clusterIndex1D]
        Z0, Y0, X0 = clusterIndex1D

        #### delete dataCLuster to recollect the memory
        del dataCluster
        gc.collect()
        ####



        fitsZero = np.zeros_like(dataCO, dtype=np.float32)
        fitsZero = fitsZero + noiseMask

        noiseMaskZero  = np.zeros_like(dataCO, dtype=np.int)

        noiseData= dataCO.copy()
        noiseData[clusterIndex1D] = np.nan #all singla regions are masked

        yxProj=np.zeros_like(dataCO[0,:,:],dtype=np.int )
        zxProj=np.zeros_like(dataCO[:,0,:],dtype=np.int )
        zyProj=np.zeros_like(dataCO[:,:,0],dtype=np.int )


        # print cloudTB.colnames
        for eachC in cloudTB:
            cloudID = eachC["_idx"]
            saveName = "cloud{}cube.fits".format(cloudID)
            saveNameNoise = "cloud{}cube_noise.fits".format(cloudID)

            cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, cloudID)
            fitsZero[cloudIndex] = dataCO[cloudIndex]

            cloudZ0, cloudY0, cloudX0 = cloudIndex

            minZ = np.min(cloudZ0)
            maxZ = np.max(cloudZ0)

            minY = np.min(cloudY0)
            maxY = np.max(cloudY0)

            minX = np.min(cloudX0)
            maxX = np.max(cloudX0)

            cropWCS = wcsCloud[minZ:maxZ + 1, minY:maxY + 1, minX:maxX + 1]
            cropData = fitsZero[minZ:maxZ + 1, minY:maxY + 1, minX:maxX + 1]
            fits.writeto(os.path.join(savePath,saveName), cropData, header=cropWCS.to_header(), overwrite=True)


            fitsZero[cloudIndex] = 1 # rest to maskfits


            ####  deal Z axis
            xyIndex= (Y0, X0)
            yxProj[xyIndex] = 1
            spectraIndicesYX = np.where(yxProj==1)

            for i in range( len(spectraIndicesYX[0] ) ):

                spectraYindex = spectraIndicesYX[0][i]
                spectraXindex = spectraIndicesYX[1][i]


                spectraMask = fitsZero[:,  spectraYindex,  spectraXindex  ]
                expandMaskV = self.expandMask( spectraMask )
                noiseMaskZero[ : ,  spectraYindex, spectraXindex ] =  noiseMaskZero[ : ,  spectraYindex, spectraXindex ]  + expandMaskV



            ####  deal Y axis
            zxIndex= (Z0, X0)
            zxProj[zxIndex] = 1
            spectraIndicesZX = np.where( zxProj==1 )


            for i in range( len(spectraIndicesZX[0] )):
                spectraZindex = spectraIndicesZX[0][i]
                spectraXindex = spectraIndicesZX[1][i]

                spectraMask = fitsZero[spectraZindex,  :,  spectraXindex  ]
                expandMaskV = self.expandMask( spectraMask )
                noiseMaskZero[ spectraZindex,  :,  spectraXindex ] = noiseMaskZero[ spectraZindex,  :,  spectraXindex ] + expandMaskV



            ####  deal X axis
            zyIndex= (Z0, Y0)
            zyProj[zyIndex] = 1
            spectraIndicesZY = np.where( zyProj==1 )


            for i in range( len(  spectraIndicesZY[0] )):
                spectraZindex = spectraIndicesZY[0][i]
                spectraYindex = spectraIndicesZY[1][i]

                spectraMask = fitsZero[spectraZindex, spectraYindex,  :  ]
                expandMaskV = self.expandMask( spectraMask )
                noiseMaskZero[ spectraZindex, spectraYindex,  : ] = noiseMaskZero[ spectraZindex, spectraYindex,  : ] + expandMaskV




            ###crop the expanded mask regions

            noiseMaskZero[cloudIndex] =0
            noiseIndices = np.where(noiseMaskZero>0)

            noiseZ0, noiseY0, noiseX0 = noiseIndices

            minZnoise = np.min(noiseZ0)
            maxZnoise = np.max(noiseZ0)

            minYnoise = np.min(noiseY0)
            maxYnoise = np.max(noiseY0)

            minXnoise = np.min(noiseX0)
            maxXnoise = np.max(noiseX0)

            cropWCSnoise = wcsCloud[minZnoise:maxZnoise + 1, minYnoise:maxYnoise + 1, minXnoise:maxXnoise + 1]
            cropDataNoise = noiseData[minZnoise:maxZnoise + 1, minYnoise:maxYnoise + 1, minXnoise:maxXnoise + 1]
            cropDataNoise= cropDataNoise.copy()

            cropNoiseMask =  noiseMaskZero[minZnoise:maxZnoise + 1, minYnoise:maxYnoise + 1, minXnoise:maxXnoise + 1]
            cropDataNoise[cropNoiseMask==0] = np.nan


            fits.writeto(os.path.join(savePath,saveNameNoise), cropDataNoise, header=cropWCSnoise.to_header(), overwrite=True)
            del cropDataNoise
            gc.collect( )



            ### initialize
            noiseMaskZero[:] = 0
            fitsZero[:] = 0

            yxProj[:] = 0
            zxProj[:] = 0
            zyProj[:] = 0

        print("Cloud fits writing done!")


    def expandMask(self, spectraMask ):
        """
        extend the range of mask, the mask is actually the mask of single, expand this maks to close regions
        :param range:
        :param size:
        :return:
        """
        signalMask = np.ma.masked_array(spectraMask,mask=spectraMask) #masked are regions
        length = len(spectraMask)

        signalRegions = np.ma.clump_masked( signalMask )

        for eachRegion in signalRegions:

            startI = eachRegion.start
            endI= eachRegion.stop


            sizeRegion = endI-startI


            ##### back mask

            backEnd= min([endI+sizeRegion,length-1] )
            signalMask[endI:backEnd]=np.ma.masked

            #### fore mask
            foreStar=max([0, startI-sizeRegion] )
            signalMask[foreStar:startI]=np.ma.masked


        return signalMask.mask*1









    def getCloudIDByRow(self, eachC):

        if "_idx" in eachC.colnames:
            return eachC["_idx"]
        else:
            cloudName = eachC["sourceName"]
            id = cloudName.split("Cloud")[-1]
            return int(id)

    def removeKM34(self):
        """
        Remove bad channels around 34 km/s
        :return:
        """
        ####

        #### particularly for cloud in the third Galactic quadrant

        ####

        ####
        ####

        # this is only a rough

    def pipeLine(self, rawCOFITS=None, outPath=None, cutoff=2, minPts=4, contype=1, rmsFITS=None, rmsMean=None,splitN=1,getIntFITS=False,getCubeFITS=False):
        """
        :return:
        """

        if rawCOFITS is not None:
            self.rawCOFITS = rawCOFITS
        if rmsFITS is not None:
            self.rmsFITS = rmsFITS
        if rmsMean is not None:
            self.averageRMS = rmsMean
        if outPath is not None:
            self.processPath = outPath

        self.setDBSCANParameters(cutoff_sigma=cutoff, minPts=minPts, connectivity=contype)

        self.computeDBSCAN(splitN= splitN )
        self.getDBSCANCatlog(doClean=True)
        self.produceCleanFITS()
        if getIntFITS:
            self.produceCloudIntFITS()

        if getCubeFITS:
            self.produceIndividualClouds()
    def getLrangeList(self, lRange, splitN, overLap):
        """
        get a list of lRanges
        :return:
        """
        lRangeEdges = np.linspace(min(lRange), max(lRange), splitN + 1)

        N = len(lRangeEdges)

        lList = []

        for i in range(N - 1):

            if i == 0:
                lList.append([lRangeEdges[i], lRangeEdges[i + 1] + overLap / 2.])
                continue
            if i == N - 2:
                lList.append([lRangeEdges[i] - overLap / 2., lRangeEdges[i + 1]])
                continue

            lList.append([lRangeEdges[i] - overLap / 2., lRangeEdges[i + 1] + overLap / 2.])

        return lList

    def mosaicLabel(self, rawCOFITS, rmsFITS, processPath, splitN=3, overlapL=1):
        """
        #split the rawCOFITS into many smaller regions, do DBSCAN speratelay and then merge then togher, need to star from left to right,
        because the index in the right is used
        :param rawCOFITS:
        :param rmsFITS:
        :return:

        """
        lRange, Brange, Vrange = doFITS.getLBVRange(rawCOFITS)

        lList = self.getLrangeList(lRange, splitN, overlapL)
        lList = lList[::-1]  # from left to rights
        ####

        ###

        ####
        # crop the fits along the velocity

        cropListRaw = []
        rmsList = []

        ####
        baseNameRaw = os.path.basename(rawCOFITS)
        baseNameRMS = os.path.basename(rmsFITS)

        for i in range(splitN):
            cropListRaw.append(os.path.join(processPath, "split_{}_".format(i) + baseNameRaw))
            rmsList.append(os.path.join(processPath, "split_{}_".format(i) + baseNameRMS))

        for i in range(splitN):
            lRnage = lList[i]

            doFITS.cropFITS(rawCOFITS, Lrange=lRnage, outFITS=cropListRaw[i], overWrite=True)
            doFITS.cropFITS2D(rmsFITS, Lrange=lRnage, outFITS=rmsList[i], overWrite=True)

        ####
        # do dbscan for each

        labelFITSList = []
        coreFITSList = []
        for i in range(splitN):
            self.rawCOFITS = cropListRaw[i]
            self.rmsFITS = rmsList[i]
            self.processPath = processPath
            self.saveCore = True

            labelFITSi = self.computeDBSCAN()
            labelFITSList.append(labelFITSi)
            coreFITSList.append(self.coreFITS)

        leftFITSRaw = labelFITSList[0]
        leftFITSCore = coreFITSList[0]

        for i in range(splitN - 1):  # merge

            if i == splitN - 1 - 1:
                mergei = os.path.join(processPath, "merge_All_rawlabel.fits")
                mergeCore = os.path.join(processPath, "merge_All_corelabel.fits")

            else:
                mergei = os.path.join(processPath, "merge_{}_rawlabel.fits".format(i))
                mergeCore = os.path.join(processPath, "merge_{}_corelabel.fits".format(i))

            self.getMergeData(leftFITSRaw, labelFITSList[i + 1], saveName=mergei)
            leftFITSRaw = mergei

            self.getMergeData(leftFITSCore, coreFITSList[i + 1], saveName=mergeCore)
            leftFITSCore = mergeCore

            # self.mergeTwoLabelFITS(  leftFITS , labelFITSList[i+1], outFITSName = mergei )
        ####

        s = generate_binary_structure(3, self.connectivity)

        labelFITSData, headlabel = doFITS.readFITS(mergei)

        dataCore, headCore = doFITS.readFITS(mergeCore)

        dataCore = dataCore > 0
        gc.collect()
        labeled_core, num_features = label(dataCore, structure=s)
        gc.collect()
        labeled_core = dilation(labeled_core, s)
        gc.collect()
        labeled_core[labelFITSData == 0] = 0
        gc.collect()
        fits.writeto("mergeTestFinale.fits", labeled_core, header=headlabel, overwrite=True)

    def mergeTwoLabelFITS(self, fits1, fits2, outFITSName=None):
        """
        :return:

        mergeTwo labe FITS, by default, the overlapping area is along  the Galactic plane
        # the left fits is the main fits
        """
        print(fits1, fits2)

        #### getLBVRange
        l1, b1, v1 = doFITS.getLBVRange(fits1)
        l2, b2, v2 = doFITS.getLBVRange(fits2)

        if np.max(l1) < np.max(l2):
            fits1, fits2 = fits2, fits1
            l1, b1, v1 = doFITS.getLBVRange(fits1)
            l2, b2, v2 = doFITS.getLBVRange(fits2)

        ########
        breakL = np.mean([min(l1), max(l2)])
        dL = abs(min(l1) - max(l2))
        mergeLRange = [breakL + dL / 4, breakL - dL / 4]

        # check if the data in the range are equal
        fits1Crop = "./tmp/merge1Crop.fits"
        fits2Crop = "./tmp/merge2Crop.fits"

        doFITS.cropFITS(fits1, Lrange=mergeLRange, outFITS=fits1Crop, overWrite=True)
        doFITS.cropFITS(fits2, Lrange=mergeLRange, outFITS=fits2Crop, overWrite=True)

        data1, head1 = doFITS.readFITS(fits1Crop)
        data2, head2 = doFITS.readFITS(fits2Crop)

        maskData1 = data1 > 0
        maskData1 = maskData1 + 0

        maskData2 = data2 > 0
        maskData2 = maskData2 + 0

        if not np.sum(maskData2 - maskData1) == 0:
            print("DBSCAN results between common area are not equal, exist")

        ## start to relabel the fits
        Part1 = "./tmp/part1Merge.fits"
        Part2 = "./tmp/part2Merge.fits"
        doFITS.cropFITS(fits1, Lrange=[max(l1), breakL - dL / 4], outFITS=Part1, overWrite=True)
        doFITS.cropFITS(fits2, Lrange=[min(l2), breakL + dL / 4], outFITS=Part2, overWrite=True)

        dataPart1, headPart1 = doFITS.readFITS(Part1)
        dataPart2, headPart2 = doFITS.readFITS(Part2)
        ###

        # data1,data2, should be the same

        ####
        maxPart1 = np.max(dataPart1)
        dataPart2 = dataPart2 + maxPart1
        dataPart2[dataPart2 == maxPart1] = 0

        data2 = data2 + maxPart1
        data2[data2 == maxPart1] = 0
        #### #
        #### fruther modify part2

        indicesOverlap1 = np.where(data1 > 0)
        overlappingData1 = data1[indicesOverlap1]
        overLappingCloud1 = np.unique(overlappingData1)
        overZ1, overY1, overX1 = indicesOverlap1

        indicesOverlap2 = np.where(data2 > 0)
        overlappingData2 = data2[indicesOverlap2]
        overLappingCloud2 = np.unique(overlappingData2)
        overZ2, overY2, overX2 = indicesOverlap2

        indicesPart2 = np.where(dataPart2 > 0)
        part2Z, part2Y, part2X = indicesPart2
        part2Values = dataPart2[indicesPart2]

        #### clould be very large....
        indicesPart1 = np.where(dataPart1 > 0)
        part1Z, part1Y, part1X = indicesPart1
        part1Values = dataPart1[indicesPart1]

        #### first, check multiplicity

        for eachMC2 in overLappingCloud2:
            # index in mergeData
            overlapPart2Indices = self.getIndices(overZ2, overY2, overX2, overlappingData2, eachMC2)

            overlap1Values = data1[overlapPart2Indices]

            if len(overlap1Values) >= 2:  # two values
                print(overlap1Values, " corresponds ", eachMC2, "in data 2")

                # need to rewrite the

        ##########

        for eachMC in overLappingCloud1:
            # index in mergeData

            part1Indices = self.getIndices(overZ1, overY1, overX1, overlappingData1, eachMC)

            MC12 = data2[part1Indices]  # to check if the values are the same

            correspondMCs = np.unique(MC12)

            existingMCIndex = correspondMCs  # np.unique(indexPart2)

            if 0 in existingMCIndex:
                print("Wrong, uncompatible fits")


            for cloudParat2 in existingMCIndex:
                indicesDataPart2 = self.getIndices(part2Z, part2Y, part2X, part2Values, cloudParat2)

                dataPart2[indicesDataPart2] = eachMC

        fits.writeto(Part2, dataPart2, header=headPart2, overwrite=True)

        ################
        ####
        self.getMergeData(Part1, Part2, breakL, saveName=outFITSName)

        ######
        # self.labelFITSName= labelFITS
        # self.getCatFromLabelArray(doClean= True )

    def getMergeData(self, fits1, fits2, breakL=None, saveName=None):
        ####
        """

        :param fits1:
        :param fits2:
        :param breakL:
        :return:
        """

        l1, b1, v1 = doFITS.getLBVRange(fits1)
        l2, b2, v2 = doFITS.getLBVRange(fits2)

        if np.max(l1) < np.max(l2):
            fits1, fits2 = fits2, fits1
            l1, b1, v1 = doFITS.getLBVRange(fits1)
            l2, b2, v2 = doFITS.getLBVRange(fits2)

        ########
        breakL = np.mean([min(l1), max(l2)])

        Part1 = "./tmp/part1ForMerging.fits"
        Part2 = "./tmp/part2ForMerging.fits"

        doFITS.cropFITS(fits1, Lrange=[max(l1), breakL], outFITS=Part1, overWrite=True)
        doFITS.cropFITS(fits2, Lrange=[min(l2), breakL], outFITS=Part2, overWrite=True)

        # one column is overlapping, removing it

        dataPart1, head1 = doFITS.readFITS(Part1)
        dataPart2, head2 = doFITS.readFITS(Part2)

        maxPart1 = np.max(dataPart1)
        dataPart2 = dataPart2 + maxPart1
        dataPart2[dataPart2 == maxPart1] = 0

        Nz1, Ny1, Nx1 = dataPart1.shape

        wcs1 = WCS(head1, naxis=2)
        wcs2 = WCS(head2, naxis=2)
        l1Last, _ = wcs1.wcs_pix2world(Nx1 - 1, Ny1 / 2, 0)
        startPart2Index, _ = wcs2.wcs_world2pix(l1Last, np.mean(b2), 0)

        if startPart2Index < 0:
            print("---wrong...existing...")
            return

        startPart2Index = int(round(startPart2Index))  # cutposition of dataPart2
        dataPart2 = dataPart2[:, :, startPart2Index + 1:]

        newData = np.concatenate((dataPart1, dataPart2), axis=2)

        # newData[:,:,Nx1:]=0
        # return  newData

        if saveName is None:
            saveName = "mergedFITS.fits"

        #
        # s = generate_binary_structure(3, self.connectivity )

        # newData[newData>0] =1

        # labeled_core, num_features = label(newData,    structure=s)  # first label core, then expand, otherwise, the expanding would wrongly connected

        fits.writeto(saveName, newData, header=head1, overwrite=True)

        return saveName

    def ZZZ(self):
        pass

