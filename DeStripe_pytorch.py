import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import ndimage
import torch
import torch.fft as fft
import tifffile
from torch.nn import functional as F
import scipy
import torch.nn as nn
import torchvision
import copy
import skimage.io as ioo
import tqdm
from network import DeStripeModel, Loss, GuidedFilterHR_fast, GuidedFilterHR, GuidedFilterLoss

class DeStripe:
    def __init__(self, data_path, sample_name, isVertical = True, angleOffset = [0], isImageSequence = False, filter_keyword = [],
                 losseps = 10, qr = 0.5, resampleRatio = 2, KGF = 29, KGFh = 29, HKs = .5, sampling_in_MSEloss = 2, isotropic_hessian = True, lambda_tv = 1, lambda_hessian = 1,
                 inc = 16, n_epochs = 300, deg = 29, Nneighbors = 16, _display = True, fast_GF = False, require_global_correction = True, mask_name = None):
        """
        data_path (str): file path for volume input
        sample_name (str): volume's name, could be a .tif, .tiff, .etc (for example, destriping_sample.tiff) or image sequence, i.e., a folder name
        isVertical (boolean): direction of the stripes. True by default
        angleOffset (list): a list of angles in degree, data range for each angle is [-90, 90]. For example [-10, 0, 10] for ultramicroscope. [0] by default
        isImageSequence (boolean): True for image sequence input, False for volumetric input. False by default.
        filter_keyword (list): a list of str (if applicable). if input is image sequence, only images that contain all the filter_keyword in name are gonna be processed, [] by default
        losseps (float): eps in loss. data range [0.1, inf). 10 by default
        qr (float): a threhold. data range [0.5, inf). 0.5 by default
        resampleRatio (int): downsample ratio, data range [1, inf), 2 by default
        KGF (int): kernel size for guided filter during training. must be odd. 29 by default
        KGFh (int): kernel size for guided filter during inference. must be odd. 29 by default
        HKs (float): sigma to generate hessian kernel. data range [0.5, 1.5]. 0.5 by default
        sampling_in_MSEloss (int): downsampling when calculating MSE. data range [1, inf). 2 by default
        isotropic_hessian (boolean): True by default
        lambda_tv (float): trade-off parameter of total variation in loss. data range (0, inf). 1 by default
        lambda_hessian (float): trade-off parameter of hessian in loss. data range (0, inf). 1 by default
        inc (int): latent neuron numbers in NN. power of 2. 16 by default
        n_epochs (int): total epochs for training. data range [1, inf). 300 by default
        deg (float): angle in degree to generate wedge-like mask. data range (0, 90). 29 by default
        Nneighbors (int): data range [1, 32], 16 by default
        _display  (boolean): display result or not after processing every slice. True by default
        fast_GF (boolean): methods used for composing high-res result, False by default
        require_global_correction (boolean): True by default
        mask_name: mask's name (if applicable) could be a XX.tif, XX.tiff, .etc, None by default
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_name = os.path.join(data_path, sample_name)
        self.angleOffset, self._isVertical, self.isImageSequence, self._filter = angleOffset, isVertical, isImageSequence, filter_keyword
        self.mask_name = os.path.join(data_path, mask_name) if mask_name != None else mask_name
        self.KGFh = KGFh
        self.KGF = KGF
        self._display = _display
        qr = [qr]
        self.qr = qr
        self.losseps = losseps
        self.HKs = HKs
        self.lambda_tv = lambda_tv
        self.lambda_hessian = lambda_hessian
        self.inc = inc
        self.n_epochs = n_epochs
        self.deg = deg
        self.resampleRatio = [resampleRatio, resampleRatio]
        self.Nneighbors = Nneighbors
        self.fast_GF = fast_GF
        self.f = isotropic_hessian
        self.require_global_correction = require_global_correction
        self.sampling = sampling_in_MSEloss
        if self.isImageSequence:
            self.filename = os.listdir(self.sample_name)
            tmp = []
            for f in self.filename:
                if np.prod([c in f for c in self._filter]): tmp.append(f)
            if len(self._filter) != 0: self.filename = tmp
            self.filename.sort()
        self.m, self.n = self.tiffread(self.sample_name, 0).shape if self.isImageSequence == False else self.tiffread(self.sample_name+"/"+self.filename[0], 0).shape
        self.md, self.nd = self.m//self.resampleRatio[0]//2*2+1, self.n//self.resampleRatio[1]//2*2+1
        angleMask = np.stack([self.WedgeMask(self.md if self._isVertical == True else self.nd, self.nd if self._isVertical == True else self.md, Angle = angle, deg = self.deg) for angle in self.angleOffset], 0)
        angleMask = angleMask.reshape(angleMask.shape[0], -1)[:, :self.md*self.nd//2]
        hier_mask = np.where(angleMask == 1)[1]
        hier_ind = np.argsort(np.concatenate([np.where(angleMask.reshape(-1) == index)[0] for index in range(2)]))
        NI = self.NeighborSampling(self.md, self.nd, k_neighbor = Nneighbors) if self._isVertical == True else self.NeighborSampling(self.nd, self.md, k_neighbor = Nneighbors)
        NI = np.concatenate([NI[hier_mask == 0, 1:Nneighbors+1].T for hier_mask in angleMask], 1)
        self.NI, self.hier_mask, self.hier_ind = torch.from_numpy(NI).to(self.device), torch.from_numpy(hier_mask).to(self.device), torch.from_numpy(hier_ind).to(self.device)
        self.GuidedFilterLoss = GuidedFilterLoss(rx = self.KGF, ry = self.KGF, eps = losseps)
        if fast_GF: self.GuidedFilterHR = GuidedFilterHR_fast(rx = self.KGFh, ry = 0, angleList = angleOffset, eps = 1e-9).to(self.device)
        else: self.GuidedFilterHR = GuidedFilterHR(rX = [self.KGFh*2+1, self.KGFh], rY = [0, 0],
                                                   m = self.m if self._isVertical == True else self.n, n = self.n if self._isVertical == True else self.m, Angle = self.angleOffset)

    def NeighborSampling(self, m, n, k_neighbor):
        NI = np.zeros((m*n, k_neighbor))
        grid_x, grid_y = np.meshgrid(np.linspace(1, m, m), np.linspace(1, n, n), indexing = 'ij')
        grid_x, grid_y = grid_x-math.floor(m/2)-1, grid_y-math.floor(n/2)-1
        grid_x, grid_y = grid_x.reshape(-1)**2, grid_y.reshape(-1)**2
        ring_radius, index = 0, 0
        while 1:
            if ring_radius != 0: Norms1 = grid_y/(ring_radius**2) + grid_x/((ring_radius/n*m)**2)
            ring_radius = ring_radius + 5
            Norms2 = grid_y/(ring_radius**2) + grid_x/((ring_radius/n*m)**2)
            if ring_radius == 5: ind = np.setdiff1d(np.where(Norms2<=1)[0], np.where(Norms2==0)[0])
            elif np.where(Norms2>1)[0].shape[0] == 0: ind = np.where(Norms1>1)[0]
            else: ind = np.setdiff1d(np.where(Norms2<=1)[0], np.where(Norms1<=1)[0])
            indc = np.random.randint(len(ind), size=len(ind)*k_neighbor)
            NI[ind, :] = ind[indc].reshape(-1, k_neighbor)
            index = index+1
            if np.where(Norms2>1)[0].shape[0] == 0:
                zero_freq = (m*n)//2
                NI = NI[:zero_freq, :]
                NI[NI>zero_freq] = 2*zero_freq-NI[NI>zero_freq]
                return np.concatenate((np.linspace(0, NI.shape[0]-1, NI.shape[0])[:, np.newaxis], NI), axis = 1).astype(np.int32)

    def WedgeMask(self, md, nd, Angle, deg):
        Xv, Yv = np.meshgrid(np.linspace(0, nd, nd+1), np.linspace(0, md, md+1))
        tmp = np.arctan2(Xv, Yv)
        tmp = np.hstack((np.flip(tmp[:, 1:], 1), tmp))
        tmp = np.vstack((np.flip(tmp[1:, :], 0), tmp))
        if Angle != 0:tmp = ndimage.rotate(tmp, Angle, reshape = False)
        a = tmp[md-md//2:md+md//2+1, nd-nd//2:nd+nd//2+1]
        tmp = Xv**2 + Yv**2
        tmp = np.hstack((np.flip(tmp[:, 1:], 1), tmp))
        tmp = np.vstack((np.flip(tmp[1:, :], 0), tmp))
        if Angle != 0:tmp = ndimage.rotate(tmp, Angle, reshape = False)
        b = tmp[md-md//2:md+md//2+1, nd-nd//2:nd+nd//2+1]
        return ((a <= math.pi/180*(90 - deg)) * (b > 18)) != 0

    def tiffread(self, path, frame_index, _is_image = True):
        img = Image.open(path)
        if frame_index is not None: img.seek(frame_index)
        img = np.array(img)
        if _is_image: img[img == 0] = 1
        return img

    def train(self):
        fileList = self.filename if self.isImageSequence else np.arange(Image.open(self.sample_name).n_frames).tolist()
        for i in range(len(self.qr)): locals()["result"+str(i)] = np.zeros((len(fileList), self.m, self.n))
        for i in range(len(self.qr)): locals()["mean"+str(i)] = np.zeros(len(fileList))
        for i, s_ in enumerate(fileList):
            print("Processing No. {} slice ({} in total): ".format(i+1, len(fileList)))
            O = np.log10(self.tiffread(self.sample_name+"/"+s_, None)) if self.isImageSequence else np.log10(self.tiffread(self.sample_name, s_))
            if self.mask_name == None: map = np.zeros(O.shape)
            else: map = self.tiffread(self.mask_name, fileList.index(s_), _is_image = False)
            if self._isVertical == False: O, map = O.T, map.T
            X = torch.from_numpy(O[None, None]).float().to(self.device)
            map = torch.from_numpy(map[None, None]).float().to(self.device)
            Xd = F.interpolate(X, size = (self.md if self._isVertical else self.nd, self.nd if self._isVertical else self.md), align_corners = True, mode = "bilinear")
            map = F.interpolate(map, size = (self.md if self._isVertical else self.nd, self.nd if self._isVertical else self.md), align_corners = True, mode = "bilinear")
            map = map > 128
            Xf = fft.fftshift(fft.fft2(Xd)).reshape(-1)[:Xd.numel()//2, None]
            smoothedtarget = self.GuidedFilterLoss(Xd, Xd)
            model = DeStripeModel(Angle = self.angleOffset,
                                  hier_mask = self.hier_mask,
                                  hier_ind = self.hier_ind,
                                  NI = self.NI,
                                  m = self.md if self._isVertical == True else self.nd,
                                  n = self.nd if self._isVertical == True else self.md,
                                  KS = self.KGF,
                                  Nneighbors = self.Nneighbors, inc = self.inc,
                                  device = self.device).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
            loss = Loss(self.HKs, self.lambda_tv, self.lambda_hessian, self.sampling, self.f, self.md if self._isVertical else self.nd, self.nd if self._isVertical else self.md, self.angleOffset, self.KGF, self.losseps, self.device).to(self.device)
            for epoch in tqdm.tqdm(range(self.n_epochs), leave = False):
                optimizer.zero_grad()
                outputGNN, outputLR = model(Xd, Xf)
                l = loss(outputGNN, outputLR, smoothedtarget, Xd, map) #Xd, X
                l.backward()
                optimizer.step()
            with torch.no_grad():
                m, n = X.shape[-2:]
                outputGNN = F.interpolate(outputGNN, size = (m, n), mode = "bilinear", align_corners = True)
                if self.fast_GF == False:
                    for index, qr in enumerate(self.qr): locals()["X"+str(index)] = 10**self.GuidedFilterHR(X, outputGNN, r = qr).cpu().data.numpy()[0, 0]
                else: locals()["X"+str(0)] = 10**self.GuidedFilterHR(X, outputGNN, X).cpu().data.numpy()[0, 0]
            if self._display:
                fig = plt.figure(dpi = 300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(locals()["X0"] if self._isVertical else locals()["X0"].T, vmin = 10**O.min(), vmax = 10**O.max(), cmap = "gray")
                ax.set_title("output", fontsize = 8, pad = 1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(10**O if self._isVertical else 10**O.T, vmin = 10**O.min(), vmax = 10**O.max(), cmap = "gray")
                ax.set_title("input", fontsize = 8, pad = 1)
                plt.axis("off")
                plt.show()
            if self._isVertical == False:
                for index in range(len(self.qr)): locals()["X"+str(index)] = locals()["X"+str(index)].T
            for index in range(len(self.qr)):
                locals()["result"+str(index)][i] = locals()["X"+str(index)]
                locals()["mean"+str(index)][i] = np.mean(locals()["X"+str(index)])
        if self.require_global_correction and (len(fileList) != 1):
            print("global correcting...")
            for i in range(len(self.qr)):
                locals()["means"+str(i)] = scipy.signal.savgol_filter(locals()["mean"+str(i)], min(21, len(locals()["mean"+str(i)])), 1)
                locals()["result"+str(i)][:] = locals()["result"+str(i)] - locals()["mean"+str(i)][:, None, None] + locals()["means"+str(i)][:, None, None]
        for i in range(len(self.qr)): locals()["result"+str(i)] = np.clip(locals()["result"+str(i)], 0, 65535).astype(np.uint16)
        name, ext = os.path.splitext(self.sample_name)
        fpath = name.rstrip(ext) + "+RESULT"
        if len(self._filter) != 0:
            subpath = ""
            for fi in self._filter: subpath = subpath + fi
        else: subpath = "all"
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        if not os.path.exists(fpath+"/"+subpath):
            os.makedirs(fpath+"/"+subpath)
        for qr in self.qr:
            locals()[fpath+str(qr)] = fpath + "/" + subpath + "/" + "GFKernel{}".format(self.KGFh) + "_GFLossKernel{}".format(self.KGF) +\
                                      "_qr{}".format(qr) + "_HKs{}".format(self.HKs) + "_losseps{}".format(self.losseps) +\
                                      "_lambdatv{}".format(self.lambda_tv) + "_lambdahessian{}".format(self.lambda_hessian) +\
                                      ("_withGlobalCorrection" if self.require_global_correction else "_withoutGlobalCorrection") +\
                                      ("_withoutMask" if self.mask_name == None else "_withMask") +\
                                      ("_fastGF" if self.fast_GF else "_nofastGF")
            if not os.path.exists(locals()[fpath+str(qr)]): os.makedirs(locals()[fpath+str(qr)])
        for (i, s_) in enumerate(tqdm.tqdm(fileList, desc = "saving: ")):
            fname = "{:03d}.tif".format(s_) if type(s_) == int else "{}.tif".format(s_)
            for index, qr in enumerate(self.qr):
                tifffile.imwrite(locals()[fpath+str(qr)] + "/" + fname, np.asarray(locals()["result"+str(index)][i]))
        for index, qr in enumerate(self.qr): del locals()["result"+str(index)]