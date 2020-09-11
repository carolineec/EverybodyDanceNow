### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model_fullts(opt):
    from .pix2pixHD_model_fullts import Pix2PixHDModel
    model = Pix2PixHDModel()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

def create_model_D(opt):
    from .pix2pixHD_model_D import Pix2PixHDModel
    model = Pix2PixHDModel()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

def create_model_flowD(opt):
    from .pix2pixHD_model_flowD import Pix2PixHDModel
    model = Pix2PixHDModel()    
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model