import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '16005'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        from configs.config_MFNet import config
    elif dataset_name == 'pst':
        from configs.config_pst900 import config
    elif dataset_name == 'nyu':
        from configs.config_nyu import config
    elif dataset_name == 'sun':
        from configs.config_sunrgbd import config
    else:
        raise ValueError('Not a valid dataset name')

    print("=======================================")
    print(config.tb_dir)
    print("=======================================")

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    # Initialize the evaluation dataset and evaluator
    val_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = ValPre()
    val_dataset = RGBXDataset(val_setting, 'val', val_pre)

    best_mean_iou = 0.0  # Track the best mean IoU for model saving
    best_epoch = 100000  # Track the epoch with the best mean IoU for model saving
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                pbar.set_description(print_str, refresh=False)
            del loss
            
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        
        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()
        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        all_dev = parse_devices(args.devices)
                        # network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(all_dev[0])
                        segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                network=model, multi_scales=config.eval_scale_array,
                                                is_flip=config.eval_flip, devices=[model.device],
                                                verbose=False, config=config,
                                                )
                        _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                    config.link_val_log_file)
                        print('mean_IoU:', mean_IoU)
                        
                        # Determine if the model performance improved
                        if mean_IoU > best_mean_iou:
                            # If the model improves, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                        else:
                            # If the model does not improve, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                        
                    model.train()
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval() 
                with torch.no_grad():
                    devices_val = [engine.local_rank] if engine.distributed else [0]
                    segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                                            network=model, multi_scales=config.eval_scale_array,
                                            is_flip=config.eval_flip, devices=[1,2,3],
                                            verbose=False, config=config,
                                            )
                    _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                config.link_val_log_file)
                    print('mean_IoU:', mean_IoU)
                    
                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()