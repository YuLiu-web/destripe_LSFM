from DeStripe_pytorch import DeStripe

data_path = "/content/drive/MyDrive/LSFM/DeStripe/Data/Single-View"
sample_name = "destriping_sample.tiff"
angleOffset = [-10, 0, 10]
isVertical = False
filter_keyword = []
isImageSequence = False
mask_name = "destriping_sample+mask.tif"
filter_keyword = []

network = DeStripe(data_path = data_path, 
                   sample_name = sample_name, 
                   mask_name = mask_name, 
                   angleOffset = angleOffset, 
                   isVertical = isVertical,
                   isImageSequence = isImageSequence, 
                   filter_keyword = filter_keyword,
                   losseps = 10,
                   qr = .5,
                   resampleRatio = 3,
                   _display = True,
                   KGF = 29,
                   KGFh = 29,
                   HKs = .5,
                   isotropic_hessian = True,
                   lambda_tv = 1,
                   lambda_hessian = 1,
                   sampling_in_MSEloss = 2,
                   deg = 29,
                   n_epochs = 300,
                   Nneighbors = 16,
                   fast_GF = False, 
                   require_global_correction = True)
network.train()

