import os
import cv2
import time
import torch
import shutil
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):

    lr_scale = (1 - float(cur_iters) / max_iters) ** power

    for param_group in optimizer.param_groups:

        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']

        if param_group.get('lr', 0) == 0:

            continue

        param_group['lr'] = param_group['initial_lr'] * lr_scale

    return base_lr * lr_scale

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)

def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'

    cfg_file = open('config.py',"r")
    cfg_lines = cfg_file.readlines()

    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)

    return writer, log_file

def logger_txt(log_file, epoch, scores):
    snapshot_name = 'ep_%d' % epoch
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%3f'%data)
    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('[')
        for key, data in scores.items():
            f.write(' '+ key+' %.2f'% data)
        f.write('\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')


def print_NWPU_summary(trainer, scores):
    f1m_l, ap_l, ar_l, mae, mse, nae, loss = scores
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')

        f.write('  [F1 %.4f Pre %.4f Rec %.4f ] [mae %.4f mse %.4f nae %.4f] [val loss %.4f]\n\n' % (f1m_l, ap_l, ar_l,mae, mse, nae,loss))

        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    print( '  [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f mse %.2f], [val loss %.4f]'\
            % (f1m_l, ap_l, ar_l, mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'],\
                                                        train_record['best_F1'],\
                                                        train_record['best_Pre'],\
                                                        train_record['best_Rec'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_nae']) )
    print( '='*50 )

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')
        f.write('  [')
        for key, data in scores.items():
            f.write(' ' +key+  ' %.3f'% data)
        f.write('\n\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print( content)
    print( '    '+ '-'*20 )
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print( best_str)
    print( '='*50 )

def update_model(trainer, scores):
    train_record = trainer.train_record
    log_file = trainer.log_txt
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d'% (epoch, trainer.i_tb)
    for key, data in scores.items():
        snapshot_name += ('_'+ key+'_%.3f'%data)

    for key, data in  scores.items():
        print(key,data)
        if data < train_record[key] :
            train_record['best_model_name'] = snapshot_name
            if log_file is not None:
                logger_txt(log_file, epoch, scores)
            to_saved_weight = trainer.model.state_dict()

            torch.save(to_saved_weight, os.path.join(trainer.exp_path, trainer.exp_name, snapshot_name + '.pth'))

        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record':train_record, 'net':trainer.model.state_dict(), 'optimizer':trainer.optimizer.state_dict(),
                    'epoch': trainer.epoch, 'i_tb':trainer.i_tb,\
                    'exp_path':trainer.exp_path, 'exp_name':trainer.exp_name}
    torch.save(latest_state, os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

    return train_record

def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

class AverageCategoryMeter(object):

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):

        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def change2map(intput_map):
    intput_map = intput_map.squeeze(0)
    vis_map = (intput_map - intput_map.min()) / (intput_map.max() - intput_map.min() + 1e-5)
    vis_map = (vis_map * 255).astype(np.uint8)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    return vis_map

def save_visual_results(data, restor_transform, save_base, iter_num, rank):
    assert (len(data)-1) % 2 == 0
    num = (len(data)-1) // 2
    h = data[0].size(2)
    w = data[0].size(3)

    batch_size = data[0].size(0)
    margin = 5

    W = w * len(data) + margin * num + 3 * margin * num
    H = h * batch_size + margin * (batch_size - 1)

    out = np.zeros((H, W, 3))

    start_h = 0

    for i in range(batch_size):
        start_w = 0
        img = cv2.cvtColor(np.array(restor_transform(data[0][i])), cv2.COLOR_RGB2BGR)
        out[start_h:start_h + h, start_w:start_w + w] = img
        start_w += w + 3 * margin

        for j in range(num):
            data_map = data[1 + j*2][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + margin

            data_map = data[1 + j*2 + 1][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + 3 * margin

        start_h += h + margin
    if not os.path.exists(save_base):
        os.makedirs(save_base, exist_ok=True)
    cv2.imwrite(os.path.join(save_base, "{}_{}_visual.jpg".format(rank, iter_num)), out)

def save_test_visual(visual_maps, imgs, scene_name, restor_transform, save_path, iter, rank):
    visual_data = [visual_maps[:, i, :, :] for i in range(visual_maps.shape[1])]
    visual_data = [torch.stack(imgs, dim=0)] + visual_data
    save_visual_results(visual_data, restor_transform, os.path.join(save_path, scene_name), iter, rank)

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow'],\
                                   gt_dict['inflow'], gt_dict['outflow']),0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]

        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]

    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2), 'MIAE':torch.zeros(0), 'MOAE':torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']

        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict, gt_dict, intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0]-inflow_cnt[:,1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])

    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])

    return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']

def local_maximum_points(sub_pre, gaussian_maximun, radius=8.):
    sub_pre = sub_pre.detach()
    _,_,h,w = sub_pre.size()
    kernel = torch.ones(3,3)/9.
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(sub_pre.device)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    sub_pre = F.conv2d(sub_pre, weight, stride=1, padding=1)

    keep = F.max_pool2d(sub_pre, (5, 5), stride=2, padding=2)
    keep = F.interpolate(keep, scale_factor=2)
    keep = (keep == sub_pre).float()
    sub_pre = keep * sub_pre

    sub_pre[sub_pre < 0.25*gaussian_maximun] = 0
    sub_pre[sub_pre > 0] = 1
    count = int(torch.sum(sub_pre).item())

    points = torch.nonzero(sub_pre)[:,[0,1,3,2]].float()
    rois = torch.zeros((points.size(0), 5)).float().to(sub_pre)
    rois[:, 0] = points[:, 0]
    rois[:, 1] = torch.clamp(points[:, 2] - radius, min=0)
    rois[:, 2] = torch.clamp(points[:, 3] - radius, min=0)
    rois[:, 3] = torch.clamp(points[:, 2] + radius, max=w)
    rois[:, 4] = torch.clamp(points[:, 3] + radius, max=h)

    pre_data = {'num': count, 'points': points, 'rois': rois}
    return pre_data
