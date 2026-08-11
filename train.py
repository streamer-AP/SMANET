import os
import torch
import torch.distributed as dist
import sys

current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import datasets
import numpy as np
from tqdm import tqdm
from misc import tools
from config import cfg
from torch import optim
from misc.utils import *
from copy import deepcopy
import torch.nn.functional as F
from model.points_from_den import *
from torch.nn import SyncBatchNorm
from misc.tools import is_main_process
from model.smanet import SMANet, remap_legacy_state_dict

class Trainer():
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        if cfg.distributed:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda", cfg.gpu)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model_without_ddp = SMANet(cfg, cfg_data)
        self.cfg_data = cfg_data

        self.model.to(self.device)

        self.val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS
        if cfg.distributed:
            sync_model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                sync_model,
                device_ids=[cfg.gpu],
                find_unused_parameters=cfg.DDP_FIND_UNUSED_PARAMETERS,
            )
            self.model_without_ddp = self.model.module

        self.train_loader, self.sampler_train, self.val_loader, self.restore_transform =\
                datasets.loading_data(cfg.DATASET, self.val_frame_intervals, cfg.distributed, is_main_process())

        new_module_params = []
        base_model_params = []

        new_module_names = {'flow_embed', 'probability_decoder'}

        for name, param in self.model_without_ddp.named_parameters():
            if not param.requires_grad:
                continue
            is_new = any(n in name for n in new_module_names)
            if is_new:
                new_module_params.append(param)
            else:
                base_model_params.append(param)

        param_groups = [
            {'params': new_module_params, 'lr': cfg.NEW_MODULE_LR, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': base_model_params, 'lr': cfg.BASE_MODULE_LR, 'weight_decay': cfg.WEIGHT_DECAY},
        ]

        self.optimizer = optim.Adam(param_groups)

        self.i_tb = 0
        self.epoch = 1
        self.num_iters = cfg.MAX_EPOCH * int(len(self.train_loader))
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE': 1e20,
                             'WRAE': 1e20, 'MIAE': 1e20, 'MOAE': 1e20, 'share_mae': 1e20, 'share_mse': 1e20}

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH, weights_only=False)
            self.model.load_state_dict(remap_legacy_state_dict(latest_state['net']), strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            # latest_state is written after an epoch's validation completes.
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            print("Finish loading resume mode")

        if cfg.PRE_TRAIN_COUNTER:
            print(f"==> Loading pre-trained weights from: {cfg.PRE_TRAIN_COUNTER}")
            checkpoint = torch.load(cfg.PRE_TRAIN_COUNTER, map_location='cpu')
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                pretrained_dict = checkpoint['net']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            pretrained_dict = remap_legacy_state_dict(pretrained_dict)

            model_dict = self.model.state_dict()
            new_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('module.') and not hasattr(self.model, 'module'):
                    k_model = k[7:]
                elif not k.startswith('module.') and hasattr(self.model, 'module'):
                    k_model = 'module.' + k
                else:
                    k_model = k

                if k_model in model_dict:
                    if model_dict[k_model].shape == v.shape:
                        new_dict[k_model] = v
                    else:
                        print(f"Warning: {k_model} shape mismatch. Pretrained: {v.shape}, Current: {model_dict[k_model].shape}. Skipping.")

            model_dict.update(new_dict)
            self.model.load_state_dict(model_dict, strict=True)
            print(f"==> Successfully loaded {len(new_dict)} layers from pre-trained model.")

        if is_main_process():
            self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd,
                                               ['exp', 'eval', 'figure', 'img', 'vis', 'output', 'exp_backup', 'pre_train_model', 'visual_results'],
                                               resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH + 1):
            self.epoch = epoch
            if cfg.distributed:
                self.sampler_train.set_epoch(epoch)
            self.timer['train time'].tic()
            smoke_finished = self.train()
            self.timer['train time'].toc(average=False)
            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            if smoke_finished:
                return

            if epoch % cfg.VAL_INTERVAL == 0 and epoch >= cfg.START_VAL:
                if cfg.distributed:
                    torch.distributed.barrier()
                self.timer['val time'].tic()
                self.validate()
                self.timer['val time'].toc(average=False)
                if is_main_process():
                    print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):
        self.model.train()
        batch_loss = {}
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            lr = adjust_learning_rate(
                self.optimizer,
                cfg.NEW_MODULE_LR,
                self.num_iters,
                self.i_tb,
                power=cfg.LR_POWER,
            )
            img, target, flow = data
            for i in range(len(target)):
                for key, data in target[i].items():
                    if torch.is_tensor(data):
                        target[i][key] = data.to(self.device)
            img = img.to(self.device)
            flow = [f.to(self.device) for f in flow]

            pre_global_den, gt_global_den, pre_share_den, gt_share_den,\
                pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, target, flow, self.epoch)

            all_loss = (
                loss_dict['global'] * cfg.GLOBAL_LOSS_WEIGHT +
                loss_dict['share'] * cfg.SHARE_LOSS_WEIGHT +
                loss_dict['in_out'] * cfg.IN_OUT_LOSS_WEIGHT +
                loss_dict['probability_outflow'] * cfg.PROBABILITY_LOSS_WEIGHT +
                loss_dict['probability_inflow'] * cfg.PROBABILITY_LOSS_WEIGHT
            )

            pre_global_cnt = pre_global_den.sum()
            gt_global_cnt = gt_global_den.sum()
            pre_share_cnt = pre_share_den.sum()
            gt_share_cnt = gt_share_den.sum()
            pre_in_out_cnt = pre_in_out_den.sum()
            gt_in_out_cnt = gt_in_out_den.sum()

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            loss_dict_reduced = reduce_dict(loss_dict)

            for k, v in loss_dict_reduced.items():
                if k not in batch_loss:
                    batch_loss[k] = AverageMeter()
                batch_loss[k].update(v.item())

            if (self.i_tb) % cfg.PRINT_FREQ == 0:
                if is_main_process():
                    self.writer.add_scalar('lr', lr, self.i_tb)
                    for k, v in loss_dict_reduced.items():
                        self.writer.add_scalar(k, v.item(), self.i_tb)
                    self.timer['iter time'].toc(average=False)

                    loss_keys = list(batch_loss.keys())
                    loss_str = ''.join([f"[loss_{key} {batch_loss[key].avg:.4f}]" for key in loss_keys])

                    print(f"[ep {self.epoch}][it {self.i_tb}]{loss_str}[{self.timer['iter time'].diff:.2f}s]")

                    if 'probability_outflow' in batch_loss:
                        probability_loss_str = (
                            f"    Transition probability -> Out: {batch_loss['probability_outflow'].avg:.4f} | "
                            f"In: {batch_loss['probability_inflow'].avg:.4f}"
                        )
                        print(probability_loss_str)

                    print('[cnt: gt: %.1f pred: %.1f max_pre: %.1f max_gt: %.1f]  ' %
                           (gt_global_cnt.item(), pre_global_cnt.item(),
                           pre_global_den.max().item() * self.cfg_data.DEN_FACTOR,
                           gt_global_den.max().item() * self.cfg_data.DEN_FACTOR))

            if cfg.TRAIN_VIS_INTERVAL and (self.i_tb) % cfg.TRAIN_VIS_INTERVAL == 0:
                save_visual_results([img, gt_global_den, pre_global_den, gt_share_den, pre_share_den,
                                     gt_in_out_den, pre_in_out_den],
                                    self.restore_transform,
                                    os.path.join(self.exp_path, self.exp_name, "training_visual"),
                                    self.i_tb,
                                    int(os.environ['RANK']) if 'RANK' in os.environ else 0)

            if cfg.SMOKE_MAX_ITERS and self.i_tb >= cfg.SMOKE_MAX_ITERS:
                if is_main_process():
                    print(f"[smoke] reached {cfg.SMOKE_MAX_ITERS} train iterations")
                return True

        return False

    def validate(self):
        torch.cuda.empty_cache()
        self.model.eval()
        rank = tools.get_rank()
        world_size = tools.get_world_size()

        all_scenes = list(enumerate(self.val_loader))
        my_scenes = all_scenes[rank::world_size]

        global_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, (scene_name, sub_valset) in my_scenes:
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + self.val_frame_intervals
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            visual_maps = []
            imgs = []
            for vi, data in enumerate(gen_tqdm, 0):
                if vi % self.val_frame_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'

                if frame_signal == 'match':
                    img, label, flow = data
                    for i in range(len(label)):
                        for key, data in label[i].items():
                            if torch.is_tensor(data):
                                label[i][key] = data.to(self.device)

                    img = img.to(self.device)
                    flow = [f.to(self.device) for f in flow]
                    with torch.no_grad():
                        b, c, h, w = img.shape
                        stride = cfg.INPUT_STRIDE
                        pad_h = (stride - h % stride) % stride
                        pad_w = (stride - w % stride) % stride
                        pad_dims = (0, pad_w, 0, pad_h)
                        img = F.pad(img, pad_dims, "constant")
                        h, w = img.size(2), img.size(3)
                        place_holder_img = torch.zeros((1, h, w), device=img.device)

                        gt_global_dot_map = torch.zeros((b, 1, h, w), device=img.device)
                        gt_in_out_dot_map = torch.zeros((b, 1, h, w), device=img.device)

                        for i in range(b):
                            points = label[i]['points'].long()
                            if points.numel() > 0:
                                gt_global_dot_map[i, 0, points[:, 1], points[:, 0]] = 1
                            if i % 2 == 0:
                                outflow_coords = points[label[i]['outflow_mask']].long()
                                if outflow_coords.numel() > 0:
                                    gt_in_out_dot_map[i, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1
                            else:
                                inflow_coords = points[label[i]['inflow_mask']].long()
                                if inflow_coords.numel() > 0:
                                    gt_in_out_dot_map[i, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1

                        gt_global_den = self.model_without_ddp.gaussian_smoother(gt_global_dot_map)
                        gt_in_out_den = self.model_without_ddp.gaussian_smoother(gt_in_out_dot_map)

                        if cfg.distributed:
                            pre_global_den, pre_share_den, pre_in_out_den =\
                                self.model.module.test_forward(img, flow)
                        else:
                            pre_global_den, pre_share_den, pre_in_out_den =\
                                self.model.test_forward(img, flow)

                        pre_global_den = pre_global_den.float()
                        pre_share_den = pre_share_den.float()
                        pre_in_out_den = pre_in_out_den.float()

                        pre_in_out_den[pre_in_out_den < 0] = 0

                        gt_global_cnt = gt_global_den[0].sum().item()
                        pre_global_cnt = pre_global_den[0].sum().item()

                        s_mae = abs(gt_global_cnt - pre_global_cnt)
                        s_mse = (gt_global_cnt - pre_global_cnt) ** 2
                        global_cnt_errors['mae'].update(s_mae)
                        global_cnt_errors['mse'].update(s_mse)

                        if vi == 0:
                            pred_dict['first_frame'] = pre_global_cnt
                            gt_dict['first_frame'] = gt_global_cnt

                        pred_dict['inflow'].append(pre_in_out_den[1].sum().item())
                        pred_dict['outflow'].append(pre_in_out_den[0].sum().item())
                        gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                        gt_dict['outflow'].append(gt_in_out_den[0].sum().item())

                        if cfg.SAVE_VAL_VISUAL and vi % self.val_frame_intervals == 0:
                            img0 = img[0]
                            gt_global_den0 = gt_global_den[0]
                            pre_global_den0 = pre_global_den[0]

                            if vi == 0:
                                gt_share_den_before = deepcopy(place_holder_img)
                                pre_share_den_before = deepcopy(place_holder_img)
                                gt_in_den = deepcopy(place_holder_img)
                                pre_in_den = deepcopy(place_holder_img)
                            else:
                                gt_share_den_before = deepcopy(place_holder_img)
                                pre_share_den_before = deepcopy(place_holder_img)
                                gt_in_den = previous_gt_in_out_den[1]
                                pre_in_den = previous_pre_in_out_den[1]

                            gt_share_den_next = deepcopy(place_holder_img)
                            pre_share_den_next = pre_share_den[0]
                            gt_out_den = gt_in_out_den[0]
                            pre_out_den = pre_in_out_den[0]

                            visual_map = torch.stack([gt_global_den0, pre_global_den0,
                                                      gt_share_den_before, pre_share_den_before,
                                                      gt_in_den, pre_in_den,
                                                      gt_share_den_next, pre_share_den_next,
                                                      gt_out_den, pre_out_den], dim=0)
                            visual_maps.append(visual_map.detach().cpu())
                            imgs.append(img0.detach().cpu())

                            previous_gt_in_out_den = gt_in_out_den
                            previous_pre_in_out_den = pre_in_out_den

                            if (vi + self.val_frame_intervals) > (len(sub_valset) - 1):
                                visual_map = torch.stack([gt_global_den[1], pre_global_den[1],
                                                          gt_share_den[1], pre_share_den[1],
                                                          gt_in_out_den[1], pre_in_out_den[1],
                                                          deepcopy(place_holder_img), deepcopy(place_holder_img),
                                                          deepcopy(place_holder_img), deepcopy(place_holder_img)], dim=0)
                                visual_maps.append(visual_map.detach().cpu())
                                imgs.append(img[1].detach().cpu())

            if cfg.SAVE_VAL_VISUAL and visual_maps:
                visual_maps = torch.stack(visual_maps, dim=0)
                save_test_visual(visual_maps, imgs, scene_name, self.restore_transform,
                                 os.path.join(self.exp_path, self.exp_name, "val_visual", scene_name),
                                 self.epoch, int(os.environ['RANK']) if cfg.distributed else 0)
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
            torch.cuda.empty_cache()

        if cfg.distributed and world_size > 1:
            all_pred_dicts = [None] * world_size
            all_gt_dicts = [None] * world_size
            dist.all_gather_object(all_pred_dicts, scenes_pred_dict)
            dist.all_gather_object(all_gt_dicts, scenes_gt_dict)

            scenes_pred_dict = sorted([d for sublist in all_pred_dicts for d in sublist], key=lambda x: x['id'])
            scenes_gt_dict = sorted([d for sublist in all_gt_dicts for d in sublist], key=lambda x: x['id'])

            metric_device = next(self.model_without_ddp.parameters()).device
            mae_sum = torch.tensor(global_cnt_errors['mae'].sum, dtype=torch.float64, device=metric_device)
            mae_count = torch.tensor(global_cnt_errors['mae'].count, dtype=torch.float64, device=metric_device)
            mse_sum = torch.tensor(global_cnt_errors['mse'].sum, dtype=torch.float64, device=metric_device)
            mse_count = torch.tensor(global_cnt_errors['mse'].count, dtype=torch.float64, device=metric_device)
            dist.all_reduce(mae_sum)
            dist.all_reduce(mae_count)
            dist.all_reduce(mse_sum)
            dist.all_reduce(mse_count)
            mae = (mae_sum / mae_count).item()
            mse = np.sqrt((mse_sum / mse_count).item())
        else:
            mae = global_cnt_errors['mae'].avg
            mse = np.sqrt(global_cnt_errors['mse'].avg)

        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, 1)

        if is_main_process():
            print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pre vs GT:', cnt_result)

            self.train_record = update_model(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})

            print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'seq_MAE': MAE, 'WRAE': WRAE, 'MIAE': MIAE, 'MOAE': MOAE})

        torch.cuda.empty_cache()

if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    import datasets
    from config import cfg
    from importlib import import_module

    tools.init_distributed_mode(cfg)
    tools.set_randomseed(cfg.SEED + tools.get_rank())

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    try:
        pwd = os.path.split(os.path.realpath(__file__))[0]
        cc_trainer = Trainer(cfg_data, pwd)
        cc_trainer.forward()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
