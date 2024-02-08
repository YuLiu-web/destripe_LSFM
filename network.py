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

class complexReLU(nn.Module):
    def __init__(self, inplace = False):
        super(complexReLU, self).__init__()
        self.relu = nn.ReLU(inplace = inplace)

    def forward(self, x):
        return self.relu(x.real) + 1j*self.relu(x.imag)

class ResLearning(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.inc, self.outc = inc, outc
        self.f = nn.Sequential(nn.Linear(inc, outc).to(torch.cfloat), complexReLU(), nn.Linear(outc, outc).to(torch.cfloat))
        if self.inc != self.outc: self.linear = nn.Linear(inc, outc).to(torch.cfloat)
        self.complexReLU = complexReLU()

    def __call__(self, x):
        inx = self.f(x)
        if self.inc == self.outc: return self.complexReLU(inx + x)
        else: return self.complexReLU(inx + self.linear(x))

class GuidedFilter(nn.Module):
    def __init__(self, rx, ry, Angle, m, n, device = "cuda", eps = 1e-9):
        super().__init__()
        self.eps = eps
        self.AngleNum = len(Angle)
        self.Angle = Angle
        kernelL = []
        for A in Angle:
            kernel = np.zeros((rx*4+1, rx*4+1), dtype = np.float32)
            if ry == 0: kernel[:, rx*2] = 1
            else: kernel[:, rx*2-ry:rx*2+ry+1] = 1
            kernel = ndimage.rotate(kernel, A, reshape = False, order = 2)[rx:3*rx+1, rx:3*rx+1]
            r, c = sum(kernel.sum(1) != 0)//2*2, sum(kernel.sum(0) != 0)//2*2
            kernelL.append(kernel[rx-r//2 : rx+r//2+1, rx-c//2: rx+c//2+1][None, None])
        c, r = max([k.shape[-2] for k in kernelL]), max([k.shape[-1] for k in kernelL])
        self.kernel = np.zeros((len(Angle), 1, c, r))
        for i, k in enumerate(kernelL): self.kernel[i, :, (c-k.shape[-2])//2:-(c-k.shape[-2])//2 if -(c-k.shape[-2])//2 != 0 else None,
                                                          (r-k.shape[-1])//2:-(r-k.shape[-1])//2 if -(r-k.shape[-1])//2 != 0 else None] = k
        self.kernel = torch.from_numpy(self.kernel).to(torch.float).to(device)
        self.pr, self.pc = self.kernel.shape[-1]//2, self.kernel.shape[-2]//2
        XN = torch.ones((1, 1, m, n)).to(device)
        self.N = [self.boxfilter(XN, self.kernel[i:i+1, ...]) for i in range(self.AngleNum)]

    def boxfilter(self, x, k):
        return torch.conv2d(F.pad(x, (self.pr, self.pr, self.pc, self.pc)), k, groups = 1)

    def __call__(self, X, y):
        for i in range(self.AngleNum):
            mean_y, mean_x = self.boxfilter(y, self.kernel[i:i+1, ...]) / self.N[i], self.boxfilter(X, self.kernel[i:i+1, ...]) / self.N[i]
            b = mean_y - mean_x
            X = X + b
        return X

class DeStripeModel(nn.Module):
    def __init__(self, Angle, hier_mask, hier_ind, NI, m, n, KS, Nneighbors = 16, inc = 16, device = "cuda"):
        super(DeStripeModel, self).__init__()
        self.hier_mask = hier_mask
        self.hier_ind = hier_ind
        self.NI = NI
        self.Angle = Angle
        self.m, self.n = m, n
        gx, gy = self.rotatableKernel(Wsize = 1, sigma = 1)
        if len(Angle) > 1:
            self.TVfftx, self.inverseTVfftx, self.TVffty, self.inverseTVffty = [], [], [], []
            for i, A in enumerate(Angle):
                self.fftnt(math.cos(-A/180*math.pi)*gx + math.sin(-A/180*math.pi)*gy, m, n)
                self.fftn(math.cos(-A/180*math.pi+math.pi/2)*gx + math.sin(-A/180*math.pi+math.pi/2)*gy, m, n)
            self.TVfftx, self.inverseTVfftx = np.concatenate(self.TVfftx, 0), np.concatenate(self.inverseTVfftx, 0)
            self.TVffty, self.inverseTVffty = np.concatenate(self.TVffty, 0), np.concatenate(self.inverseTVffty, 0)
        else:
            if Angle[0] != 0:
                self.TVfftx, self.inverseTVfftx = self.fftnt(math.cos(-Angle[0]/180*math.pi)*gx + math.sin(-Angle[0]/180*math.pi)*gy, m, n)
                self.TVffty, self.inverseTVffty = self.fftn(math.cos(-Angle[0]/180*math.pi+math.pi/2)*gx + math.sin(-Angle[0]/180*math.pi+math.pi/2)*gy, m, n)
            else:
                self.TVfftx, self.inverseTVfftx = self.fftnt(np.array([[1, -1]], dtype = np.float32), m, n)
                self.TVffty, self.inverseTVffty = self.fftn(np.array([[1], [-1]], dtype = np.float32), m, n)
        self.TVfftx = torch.from_numpy(self.TVfftx).to(torch.cfloat).to(device)
        self.TVffty = torch.from_numpy(self.TVffty).to(torch.cfloat).to(device)
        self.inverseTVfftx = torch.from_numpy(self.inverseTVfftx).to(torch.cfloat).to(device)
        self.inverseTVffty = torch.from_numpy(self.inverseTVffty).to(torch.cfloat).to(device)
        self.eigDtD = torch.div(1, torch.pow(torch.abs(self.TVfftx), 2)+torch.pow(torch.abs(self.TVffty), 2))
        self.p = ResLearning(1, inc)
        self.edgeProcess = nn.Sequential(nn.Linear(inc, inc).to(torch.cfloat), complexReLU(), nn.Linear(inc, inc).to(torch.cfloat))
        self.latentProcess = nn.Sequential(nn.Linear(inc, inc).to(torch.cfloat), complexReLU(), nn.Linear(inc, inc).to(torch.cfloat))
        self.merge = nn.Sequential(nn.Linear(inc*len(Angle), inc).to(torch.cfloat), complexReLU(), nn.Linear(inc, inc).to(torch.cfloat), complexReLU(), nn.Linear(inc, 1).to(torch.cfloat))
        self.base = nn.Sequential(nn.Linear(1, inc), nn.ReLU(), nn.Linear(inc, inc), nn.ReLU(), nn.Linear(inc, 1))
        self.GuidedFilter = GuidedFilter(rx = KS, ry = 0, m = m, n = n, Angle = Angle)
        self.w = nn.Parameter(torch.rand(NI.shape))
        self.complexReLU = complexReLU()
        self.ainput = torch.ones(1, 1).to(device)
        self.inc = inc

    def rotatableKernel(self, Wsize, sigma):
        k = np.linspace(-Wsize, Wsize, 2*Wsize+1)[None, :]
        g, gp = np.exp(-(k**2)/(2*sigma**2)), -(k/sigma) * np.exp(-(k**2)/(2*sigma**2))
        return g.T * gp, gp.T * g

    def fftnt(self, x, row, col):
        y = np.fft.fftshift(np.fft.fft2(x, s = (row, col))).reshape(-1)[:col*row//2][None, :, None]
        if len(self.Angle) > 1:
            self.TVfftx.append(y)
            self.inverseTVfftx.append(np.conj(y))
        else: return y, np.conj(y)

    def fftn(self, x, row, col):
        y = np.fft.fftshift(np.fft.fft2(x, s = (row, col))).reshape(-1)[:col*row//2][None, :, None]
        if len(self.Angle) > 1:
            self.TVffty.append(y)
            self.inverseTVffty.append(np.conj(y))
        else: return y, np.conj(y)

    def fourierResult(self, z, mean):
        return torch.abs(fft.ifft2(fft.ifftshift(torch.cat((z, self.base(self.ainput)*mean, torch.conj(torch.flip(z, [0]))), 0).reshape(1, 1, self.m, -1))))

    def forward(self, Xd, Xf):
        Xf = self.p(Xf)
        Xfcx = torch.sum(torch.einsum('knc,kn->knc', Xf[self.NI, :], self.w), 0)
        Xf_tvx = torch.cat((Xfcx, Xf[self.hier_mask, :]), 0)[self.hier_ind, :].reshape(len(self.Angle), -1, self.inc)
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for x, y, inverseTVfftx, inverseTVffty, eigDtD in zip(Xf_tvx, Xf_tvy, self.inverseTVfftx, self.inverseTVffty, self.eigDtD):
            X_fourier.append(self.complexReLU(self.latentProcess(self.complexReLU(self.edgeProcess(x) * inverseTVfftx) + self.complexReLU(self.edgeProcess(y) * inverseTVffty)) * eigDtD))
        X_fourier = self.merge(torch.cat(X_fourier, -1))
        outputGNN = self.fourierResult(X_fourier, Xd.sum())
        outputLR = self.GuidedFilter(Xd, torch.clip(outputGNN, 0, None))
        return outputGNN, outputLR

class GuidedFilterLoss(nn.Module):
    def __init__(self, rx, ry, eps = 1e-9):
        super(GuidedFilterLoss, self).__init__()
        self.rx, self.ry, self.eps = rx, ry, eps

    def diff_x(self, input, r):
        left   = input[:, :,         r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
        right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]
        output = torch.cat([left, middle, right], 2)
        return output

    def diff_y(self, input, r):
        left   = input[:, :, :,         r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
        right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]
        output = torch.cat([left, middle, right], 3)
        return output

    def boxfilter(self, input):
        return self.diff_y(self.diff_x(input.cumsum(2), self.rx).cumsum(3), self.ry)

    def forward(self, x, y):
        N = self.boxfilter(torch.ones_like(x))
        mean_x, mean_y = self.boxfilter(x) / N, self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / torch.clip(var_x, self.eps, None)
        b = mean_y - A * mean_x
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b

class Loss(nn.Module):
    def __init__(self, HKs, lambda_tv, lambda_hessian, sampling, f, md, nd, Angle, KGF, losseps, device):
        super(Loss, self).__init__()
        self.md, self.nd = md, nd
        self.angleOffset = Angle
        self.Dy = torch.from_numpy(np.array([[1], [-1]], dtype = np.float32)[None, None]).to(device)
        self.Dx = torch.from_numpy(np.array([[1, -1]], dtype = np.float32)[None, None]).to(device)
        self.DGaussxx, self.DGaussyy, self.DGaussxy = self.generateHessianKernel(HKs)
        self.DGaussxx, self.DGaussyy, self.DGaussxy = self.DGaussxx.to(device), self.DGaussyy.to(device), self.DGaussxy.to(device)
        self.GuidedFilterLoss = GuidedFilterLoss(rx = KGF, ry = KGF, eps = losseps)
        self.lambda_tv, self.lambda_hessian = lambda_tv, lambda_hessian
        self.sampling = sampling
        self.f = f

    def rotatableKernel(self, Wsize, sigma):
        k = torch.linspace(-Wsize, Wsize, 2*Wsize+1)[None, :]
        g = torch.exp(-(k**2)/(2*sigma**2))
        gp = -(k/sigma) * torch.exp(-(k**2)/(2*sigma**2))
        return g.T * gp, gp.T * g

    def generateHessianKernel(self, Sigma):
        Wsize = math.ceil(3*Sigma)
        KernelSize = 2 * (2 * Wsize + 1) - 1
        gx, gy = self.rotatableKernel(Wsize, Sigma)
        gxFFT2, gyFFT2 = fft.fft2(gx, s = (self.md, self.nd)), fft.fft2(gy, s = (self.md, self.nd))
        DGaussxx = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussxy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussyy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        for i, A in enumerate(self.angleOffset):
            a, b, c, d = math.cos(-A/180*math.pi), math.sin(-A/180*math.pi), math.cos(-A/180*math.pi+math.pi/2), math.sin(-A/180*math.pi+math.pi/2)
            DGaussxx[i] = fft.ifft2((a*gxFFT2 + b*gyFFT2) * (a*gxFFT2 + b*gyFFT2)).real[None, None, :KernelSize, :KernelSize].float()
            DGaussyy[i] = fft.ifft2((c*gxFFT2 + d*gyFFT2) * (c*gxFFT2 + d*gyFFT2)).real[None, None, :KernelSize, :KernelSize].float()
            DGaussxy[i] = fft.ifft2((c*gxFFT2 + d*gyFFT2) * (a*gxFFT2 + b*gyFFT2)).real[None, None, :KernelSize, :KernelSize].float()
        return DGaussxx, DGaussyy, DGaussxy

    def TotalVariation(self, x, target):
        return torch.abs(torch.conv2d(x, self.Dx)).sum() + torch.abs(torch.conv2d(x-target, self.Dy)).sum()

    def HessianRegularizationLoss(self, x, target):
        return torch.abs(torch.conv2d(x, self.DGaussxx)).sum() + torch.abs(torch.conv2d(x-target, self.DGaussyy)).sum() +\
               (torch.abs(torch.conv2d(x-target, self.DGaussxy)) * (2 if self.f else 1)).sum()

    def forward(self, outputGNN, outputLR, smoothedTarget, targets, map):
        mse = torch.sum(torch.abs(smoothedTarget - self.GuidedFilterLoss(outputLR, outputLR))) +\
              torch.sum((torch.abs(targets - outputLR) * map)[:, :, ::self.sampling, ::self.sampling])
        tv = 1*self.TotalVariation(outputGNN, targets) + 1*self.TotalVariation(outputLR, targets)
        hessian = 1*self.HessianRegularizationLoss(outputGNN, targets) + 1*self.HessianRegularizationLoss(outputLR, targets)
        return mse + self.lambda_tv*tv + self.lambda_hessian*hessian

class BoxFilter(nn.Module):
    def __init__(self, rx, ry, Angle):
        super(BoxFilter, self).__init__()
        self.rx, self.ry = rx, ry
        if Angle != 0:
            kernelx = torch.zeros(rx*4+1, rx*4+1)
            kernelx[:, rx*2] = 1
            kernelx = ndimage.rotate(kernelx, Angle, reshape = False, order = 1)[rx:3*rx+1, rx:3*rx+1]
            self.kernelx = torch.from_numpy(kernelx[None, None, :, :]).float().cuda()
        else: self.kernelx = torch.ones(1, 1, 2*rx+1, 2*ry+1).float().cuda()
        self.Angle = Angle

    def diff_x(self, input):
        if self.Angle != 0: return torch.conv2d(F.pad(input, (self.rx, self.rx, self.rx, self.rx), mode = "circular"), self.kernelx)
        else: return torch.conv2d(F.pad(input, (self.ry, self.ry, self.rx, self.rx), mode = "circular"), self.kernelx)

    def forward(self, x):
        return self.diff_x(x)

class GuidedFilterHR_fast(nn.Module):
    def __init__(self, rx, ry, angleList, eps = 1e-9):
        super(GuidedFilterHR_fast, self).__init__()
        self.eps = eps
        self.boxfilter = [BoxFilter(rx, ry, Angle = Angle) for Angle in angleList]
        self.N = None
        self.angleList = angleList
        self.crop = None

    def forward(self, xx, yy, hX):
        with torch.no_grad():
            if self.crop == None: self.crop = torchvision.transforms.CenterCrop(xx.size()[-2:])
            AList, bList = [], []
            for i, Angle in enumerate(self.angleList):
                x = torchvision.transforms.functional.rotate(xx, Angle, expand = True)
                y = torchvision.transforms.functional.rotate(yy, Angle, expand = True)
                h_x, w_x = x.size()[-2:]
                self.N = self.boxfilter[i](x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))
                mean_x = self.boxfilter[i](x) / self.N
                var_x = self.boxfilter[i](x * x) / self.N - mean_x * mean_x
                mean_y = self.boxfilter[i](y) / self.N
                cov_xy = self.boxfilter[i](x * y) / self.N - mean_x * mean_y
                A = cov_xy / (var_x + self.eps)
                b = mean_y - A * mean_x
                A, b = (self.boxfilter[i](A) / self.N), self.boxfilter[i](b) / self.N
                result = A * x + b
                AList.append(A)
                bList.append(b)
                xx = self.crop(torchvision.transforms.functional.rotate(result, -Angle, expand = True))
                yy = self.crop(torchvision.transforms.functional.rotate(yy, -Angle, expand = True))
            h_hrx, w_hrx = hX.size()[-2:]
            cropH = torchvision.transforms.CenterCrop((h_hrx, w_hrx))
            for i, (Angle, A, b) in enumerate(zip(self.angleList, AList, bList)):
                hXX = torchvision.transforms.functional.rotate(hX, Angle, expand = True)
                hA = F.interpolate(A, hXX.shape[-2:], mode = 'bilinear', align_corners = True)
                hb = F.interpolate(b, hXX.shape[-2:], mode = 'bilinear', align_corners = True)
                result = hA * hXX + hb
                hX = cropH(torchvision.transforms.functional.rotate(result, -Angle, expand = True))
                return hX

class GuidedFilterHR(nn.Module):
    def __init__(self, rX, rY, Angle, m, n, eps = 1e-9):
        super(GuidedFilterHR, self).__init__()
        self.eps = eps
        self.AngleNum = len(Angle)
        self.Angle = Angle
        self.PR, self.PC, self.KERNEL = [], [], []
        for rx, ry in zip(rX, rY):
            self.KERNEL.append(torch.ones((1, 1, rx*2+1, ry*2+1)).cuda())
            self.PR.append(ry)
            self.PC.append(rx)
        self.GaussianKernel = torch.tensor(np.ones((rX[1], 1)) * np.ones((1, rX[1])), dtype = torch.float32)
        self.GaussianKernel = (self.GaussianKernel / self.GaussianKernel.sum())[None, None].cuda()
        self.GaussianKernelpadding = (self.GaussianKernel.shape[-2]//2, self.GaussianKernel.shape[-1]//2)
        self.crop = torchvision.transforms.CenterCrop((m, n))
        self.K = torch.from_numpy(self.sgolay2dkernel(np.array([rX[1], 1]), np.array([2, 2])).astype(np.float32)).cuda()[None, None]

    def boxfilter(self, x, weight, k, pc, pr):
        b, c, m, n = x.shape
        img = F.pad(x, (pr, pr, pc, pc)).unfold(-2, k.shape[-2], 1).unfold(-2, k.shape[-1], 1).flatten(-2)
        return (img * weight).sum(-1)

    def sgolay2dkernel(self, window_size, order):
        n_terms = (order + 1) * (order + 2) / 2.0
        half_size = window_size // 2
        exps = []
        for row in range(order[0] + 1):
            for column in range(order[1] + 1):
                if (row + column) > max(*order): continue
                exps.append((row, column))
        indx = np.arange(-half_size[0], half_size[0]+1, dtype = np.float64)
        indy = np.arange(-half_size[1], half_size[1]+1, dtype = np.float64)
        dx = np.repeat(indx, window_size[1])
        dy = np.tile(indy, [window_size[0], 1]).reshape(window_size[0]*window_size[1],)
        A = np.empty((window_size[0]*window_size[1], len(exps)))
        for i, exp in enumerate(exps): A[:, i] = (dx**exp[0]) * (dy**exp[1])
        return np.linalg.pinv(A)[0].reshape((window_size[0], -1))

    def generateWeight(self, y, k, pc, pr, sigma):
        b, c, m, n = y.shape
        weight = torch.exp(-(F.pad(y, (pr, pr, pc, pc)).unfold(-2, k.shape[-2], 1).unfold(-2, k.shape[-1], 1).flatten(-2) - y[:, :, :, :, None])**2 / (sigma/2)**2)
        d = (torch.linspace(0, k.shape[-2]-1, k.shape[-2]) - k.shape[-2]//2)[:, None]**2 + (torch.linspace(0, k.shape[-1]-1, k.shape[-1]) - k.shape[-1]//2)[None, :]**2
        d = torch.exp(-d.flatten() / (k.numel()/4)**2)[None, None, None, None, :].repeat(b, c, m, n, 1).cuda()
        validPart = F.pad(torch.ones_like(y), (pr, pr, pc, pc)).unfold(-2, k.shape[-2], 1).unfold(-2, k.shape[-1], 1).flatten(-2)
        weight[:] = weight * d
        weight[:] = weight * validPart
        weight[:] = weight / weight.sum(-1, keepdim = True)
        return weight

    def forward(self, X, y, r):
        with torch.no_grad():
            sigma = r*(y.max() - y.min())
            for i, angle in enumerate(self.Angle):
                XX = torchvision.transforms.functional.rotate(X, angle, expand = True)
                yy = torchvision.transforms.functional.rotate(y, angle, expand = True)
                bdetail, Abase, bbase = [], [], []
                XXbase, yybase = torch.conv2d(XX, self.GaussianKernel, padding = self.GaussianKernelpadding), torch.conv2d(yy, self.GaussianKernel, padding = self.GaussianKernelpadding)
                XXdetail, yydetail = XX - XXbase, yy - yybase
                list_ = np.arange(XX.shape[-1])
                g, o = 64, 2
                list_ = [list_[i:i+g] for i in range(0, len(list_), g-o)]
                if len(list_[-1]) <= 5:
                    list_[-2] = np.arange(list_[-2][0], XX.shape[-1]).tolist()
                    del list_[-1]
                for index, i in enumerate(list_):
                    Xbase, Xdetail, ybase, ydetail = XXbase[..., i[0]:i[-1]+1], XXdetail[..., i[0]:i[-1]+1], yybase[..., i[0]:i[-1]+1], yydetail[..., i[0]:i[-1]+1]
                    weightDetail = self.generateWeight(Xdetail, self.KERNEL[0], self.PC[0], self.PR[0], sigma)
                    weightBase = self.generateWeight(Xbase, self.KERNEL[1], self.PC[1], self.PR[1], sigma)
                    bdetail.append(self.boxfilter(ydetail - Xdetail, weightDetail, self.KERNEL[0], self.PC[0], self.PR[0])[..., o//2 if index != 0 else 0:-o//2 if index != len(list_)-1 else None])
                    mean_y, mean_x = self.boxfilter(ybase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1]), self.boxfilter(Xbase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1])
                    var_x = self.boxfilter(Xbase * Xbase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1]) - mean_x * mean_x
                    cov_xy = self.boxfilter(Xbase * ybase, weightBase, self.KERNEL[1], self.PC[1], self.PR[1]) - mean_x * mean_y
                    A = cov_xy / (var_x + 1e-6)
                    b = mean_y - A * mean_x
                    bbase.append(b[..., o//2 if index != 0 else 0:-o//2 if index != len(list_)-1 else None])
                    Abase.append(A[..., o//2 if index != 0 else 0:-o//2 if index != len(list_)-1 else None])
                bdetail = torch.cat(bdetail, -1)
                Abase, bbase = torch.cat(Abase, -1), torch.cat(bbase, -1)
                #bdetail = torch.conv2d(bdetail, self.K, padding = (self.K.shape[-2]//2, self.K.shape[-1]//2))
                result = Abase * XXbase + bbase + XXdetail + bdetail
                X = self.crop(torchvision.transforms.functional.rotate(result, -angle, expand = True))
                y = self.crop(torchvision.transforms.functional.rotate(yy, -angle, expand = True))
            return X
