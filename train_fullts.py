### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_fullts
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

""" new residual model """
model = create_model_fullts(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter    
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################


        no_nexts = data['next_label'].dim() > 1 #check if has a next label (last training pair does not have a next label)

        if no_nexts:
            cond_zeros = torch.zeros(data['label'].size()).float()

            losses, generated = model(Variable(data['label']), Variable(data['next_label']), Variable(data['image']), \
                    Variable(data['next_image']), Variable(data['face_coords']), Variable(cond_zeros), infer=True)

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + (loss_dict['D_realface'] + loss_dict['D_fakeface']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_GANface']

            ############### Backward Pass ####################
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == 0:
                errors = {}
                if torch.__version__[0] == '1':
                    errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                else:
                    errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                syn = generated[0].data[0]
                inputs = torch.cat((data['label'], data['next_label']), dim=3)
                targets = torch.cat((data['image'], data['next_image']), dim=3)
                visuals = OrderedDict([('input_label', util.tensor2im(inputs[0], normalize=False)),
                                           ('synthesized_image', util.tensor2im(syn)),
                                           ('real_image', util.tensor2im(targets[0]))])
                if opt.face_generator: #display face generator on tensorboard
                    miny, maxy, minx, maxx = data['face_coords'][0]
                    res_face = generated[2].data[0]
                    syn_face = generated[1].data[0]
                    preres = generated[3].data[0]
                    visuals = OrderedDict([('input_label', util.tensor2im(inputs[0], normalize=False)),
                                           ('synthesized_image', util.tensor2im(syn)),
                                           ('synthesized_face', util.tensor2im(syn_face)),
                                           ('residual', util.tensor2im(res_face)),
                                           ('real_face', util.tensor2im(data['image'][0][:, miny:maxy, minx:maxx])),
                                           # ('pre_residual', util.tensor2im(preres)),
                                           # ('pre_residual_face', util.tensor2im(preres[:, miny:maxy, minx:maxx])),
                                           ('input_face', util.tensor2im(data['label'][0][:, miny:maxy, minx:maxx], normalize=False)),
                                           ('real_image', util.tensor2im(targets[0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch  
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        print('------------- finetuning Local + Global generators jointly -------------')
        model.module.update_fixed_params()

    ### instead of only training the face discriminator, train the entire network after certain iterations
    if (opt.niter_fix_main != 0) and (epoch == opt.niter_fix_main):
        print('------------- traing all the discriminators now and not just the face -------------')
        model.module.update_fixed_params_netD()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
