import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import triple_transforms
from network import GenerativeNetwork
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir,ReplayBuffer

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'SemiMoreGanNet'

args = {
    'iter_num': 1000000,
    'train_batch_size': 2,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'img_size_h': 512,
    'img_size_w': 1024,
    'crop_size': 512,
    'snapshot_epochs': 10000
}
transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    # triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder(transform=transform, target_transform=transform,
                        triple_transform=triple_transform, is_supervised=True)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
result_buffer = ReplayBuffer()

criterion = nn.L1Loss()
criterion_depth = nn.L1Loss()
log_path = os.path.join(ckpt_path, exp_name, "total" + '.txt')

def main():
    gs_net = GenerativeNetwork().cuda().train()
    GS_optimizer = optim.Adam([
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])
    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        gs_net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        GS_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        GS_optimizer.param_groups[0]['lr'] = 2 * args['lr']
        GS_optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(gs_net,GS_optimizer)
def train(gs_net, GS_optimizer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record = AvgMeter()
        train_L1loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()
        for i, data in enumerate(train_loader):
            GS_optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            GS_optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gts, dps,reals = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()
            GS_optimizer.zero_grad()
            GS_result, depth_pred = gs_net(inputs)
            l1loss = nn.L1Loss()
            loss_net = l1loss(GS_result, gts)
            loss_depth = l1loss(depth_pred, dps)
            loss = loss_net + loss_depth
            loss.backward()
            GS_optimizer.step()
            train_loss_record.update(loss.data, batch_size)
            train_L1loss_record.update(loss_net.data, batch_size)
            train_depth_loss_record.update(loss_depth.data, batch_size)
            curr_iter += 1
            log = '[iter %d], [Total_loss %.13f], [lr %.13f], [L1_loss %.13f], [loss_depth %.13f]' % \
                  (curr_iter, train_loss_record.avg, GS_optimizer.param_groups[1]['lr'],
                   train_L1loss_record.avg, train_depth_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % args['snapshot_epochs'] == 0:
                torch.save(gs_net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1))))
                torch.save(GS_optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1))))

            if curr_iter > args['iter_num']:
                return
if __name__ == '__main__':
    main()