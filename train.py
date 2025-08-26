import os
import sys
import json
import time
import collections

from collections import OrderedDict
from os.path import dirname, abspath

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Local imports
amber_dir = dirname(abspath(__file__))
amber_dir = dirname(abspath(__file__))
sys.path.insert(0, amber_dir)
sys.path.insert(0, os.path.join(amber_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(amber_dir, 'densevid_eval3/SODA'))

import distributed as dist_utils
import opts
from eval_utils import evaluate
from misc.utils import (
    print_alert_message,
    build_folder,
    create_logger,
    backup_envir,
    print_opt,
    set_seed
)
from data.video_dataset import PropSeqDataset, collate_fn
from amber.amber import build

def train(opt):
    set_seed(opt.seed)
    # Setup output folder & logger
    save_folder = build_folder(opt)
    logger = create_logger(save_folder, 'train.log')
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))
    if not opt.start_from:
        backup_envir(save_folder)
        logger.info('Backup environment completed!')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # Resume training if start_from is set                    
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            logger.info(f'Load info from {infos_path}')
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']
            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for name, value in prev_opt.items():
                if name not in exclude_opt:
                    vars(opt)[name] = value
                if prev_opt.get(name) != vars(opt).get(name):
                    logger.info(f'Change opt {name} : {prev_opt.get(name)} --> {vars(opt).get(name)}')

    # Dataset & Dataloader
    print_alert_message('Dataset', logger)
    train_dataset = PropSeqDataset(opt.train_caption_file,
                                    opt.visual_feature_folder,
                                    opt.dict_file, 
                                    True, # is_training
                                    'gt',
                                    opt)
    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, 
                                 False, # is_training
                                 'gt',
                                 opt)

    train_loader = DataLoader(train_dataset, 
                              batch_size=opt.batch_size, 
                              shuffle=True, num_workers=opt.nthreads, 
                              collate_fn=collate_fn, 
                              drop_last=True)    
    val_loader = DataLoader(val_dataset, 
                            batch_size=opt.batch_size_for_eval, 
                            shuffle=False, 
                            num_workers=opt.nthreads, 
                            collate_fn=collate_fn)
    
    logger.info(f'Train loader: {len(train_loader)} | Val loader: {len(val_loader)}')

    # Training states
    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    # Build model
    model, criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.to(opt.device).train()

    # Load model checkpoint
    if opt.start_from and (not opt.pretrain):
        ckpt_path = os.path.join(save_folder, f'model-{opt.start_from_mode}.pth')
        model_pth = torch.load(ckpt_path)
        logger.info(f'Loading pth from {save_folder}, iteration:{iteration}')
        model.load_state_dict(model_pth['model'])

    # Load pretrained weights
    if opt.pretrain and (not opt.start_from):
        logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        if opt.pretrain == 'encoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == 'decoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
            model.load_state_dict(decoder_pth, strict=True)
        elif opt.pretrain == 'full':
            model.load_state_dict(model_pth['model'], strict=True)
        else:
            raise ValueError("Wrong value of opt.pretrain")

    # Optimizer & Scheduler
    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)

    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.step(epoch-1)

    # Print the args for debugging
    print_opt(opt, model, logger)

    # Training loop
    print_alert_message('Start training', logger)
    loss_sum = OrderedDict()
    bad_video_num = 0
    start = time.time()

    weight_dict = criterion.weight_dict
    if dist_utils.is_main_process():
        logger.info(f'LOSS type: {list(weight_dict.keys())}')
        logger.info(f'LOSS weights: {list(weight_dict.values())}')

    # Epoch-level iteration
    while True:
        if True:
            # Scheduled sampling update
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                  opt.scheduled_sampling_max_prob)
                model.caption_head.ss_prob = opt.ss_prob
            logger.info(f'lr: {float(opt.current_lr)}')
        
        # Batch-level iteration
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            if dt is None:
                continue

            if opt.device=='cuda':
                torch.cuda.synchronize(opt.device)

            if opt.debug and (iteration + 1) % 5 == 0: # Each epoch contains less mini-batches for debugging
                iteration += 1
                break
            iteration += 1
            optimizer.zero_grad()

            # Move to device
            dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']
                ]
            
            # Forward & backward
            output, loss = model(dt, criterion) 
            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            # Track loss
            for loss_k, loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0) + loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            if opt.device=='cuda':
                torch.cuda.synchronize()

            # Logging
            losses_log_every = 6 if opt.debug else max(1, int(len(train_loader) / 10))
            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()
                logger.info(
                    f"ID {opt.id} iter {iteration} (epoch {epoch}), "
                    f"loss = {loss_sum}, time/iter = {(end - start) / losses_log_every:.3f}, "
                    f"bad_vid = {bad_video_num:.3f}"
                )
                
                tf_writer.add_scalar('lr', opt.current_lr, iteration)
                for loss_type in loss_sum.keys():
                    tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)
                # Reset for next epochs
                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache() # Frees unused GPU memory to prevent memory buildup
            
        # Evaluation & checkpointing
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):
            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }
            ckpt_path = os.path.join(
                save_folder, f'model_iter_{iteration}.pth' if opt.save_all_checkpoint else 'model-last.pth'
            )
            torch.save(saved_pth, ckpt_path)

            # Evaluation
            model.eval() # disables dropout, batch normalization updates
            result_json_path = os.path.join(save_folder, 'prediction', f'num{len(val_dataset)}_epoch{epoch}.json')
            eval_score, eval_loss = evaluate(opt, model, criterion, postprocessors, val_loader, result_json_path, logger=logger, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            
            # Compute score for best checkpoint selection
            if opt.caption_decoder_type == 'none':
                # Harmonic mean of Precision and Recall
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            elif opt.criteria_for_best_ckpt == 'dvc':
                # Mean of METEOR and soda_c scores.
                current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
            else:
                # For Paragraph captioning
                current_score = (
                    np.array(eval_score['para_METEOR']).mean() +
                    np.array(eval_score['para_CIDEr']).mean() +
                    np.array(eval_score['para_Bleu_4']).mean()
                )

            # Logging Evaluation Results
            for key, val in eval_score.items():
                # Bleu_1,2,3,4 - METEOR - ROUGE_L - CIDEr - Recall - Precision - soda_c ||| para_Bleu_1,2,3,4 - para_METEOR - para_ROUGE_L - para_CIDEr
                tf_writer.add_scalar(key, np.array(val).mean(), iteration)
            for loss_type, val in eval_loss.items():
                # loss_ce, loss_counter, loss_bbox, loss_giou, loss_self_iou, cardinality_error, total_loss
                tf_writer.add_scalar(f'eval_{loss_type}', val, iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            logger.info(f'\nValidation results of iter {iteration}:\n' +
                        '\n'.join([f'{k}:{v}' for k, v in eval_score.items()]))
            logger.info(f'\nOverall score of iter {iteration}: {current_score}')

            val_result_history[epoch] = {'eval_score': eval_score}
            logger.info(f'Save model at iter {iteration} to {ckpt_path}')

            # Save best model
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt), # Training configuration
                                      'iter': iteration, # Iteration number of the best model
                                      'epoch': best_epoch, # Epoch of the best model
                                      'best_val_score': best_val_score, # Best validation score achieved
                                      'result_json_path': result_json_path, # Path to the JSON file with validation predictions
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'], # Validation Precision  
                                      'Recall': eval_score['Recall'] # Validation Recall 
                                      }

                torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            # Save last training state
            saved_info['last'] = {'opt': vars(opt), # Training configuration
                                  'iter': iteration, # Iteration number
                                  'epoch': epoch, # Epoch
                                  'best_val_score': best_val_score,
                                  }
            saved_info['history'] = {'val_result_history': val_result_history, # Validation results per epoch
                                     'loss_history': loss_history, # Training loss history/iteration
                                     'lr_history': lr_history, # Learning rate history/iteration
                                     }

            with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                json.dump(saved_info, f)
            logger.info('Save info to info.json')

            model.train()

        epoch += 1
        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()

        # Stop criterion
        if epoch >= opt.epoch:
            if dist_utils.is_main_process():
                tf_writer.close()
                break

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    train(opt)
