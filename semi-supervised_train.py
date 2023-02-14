import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from TVLoss.TVLossL1 import TVLossL1
from ECLoss.ECLoss import DCLoss
import triple_transforms
from network import GenerativeNetwork,Discriminator
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir,ReplayBuffer

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'SemiMoreGanNet'

args = {
    'iter_num': 500000,
    'train_batch_size': 2,
    'last_iter': 0,
    'gen_lr': 5e-4,
    'dis_lr': 1e-4,
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
    gr_net = GenerativeNetwork().cuda().train()
    ds_net = Discriminator(3).cuda().train()
    dr_net = Discriminator(3).cuda().train()
    ###
    GS_optimizer = optim.Adam([
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'gen_lr': 2 * args['gen_lr']},
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'gen_lr': args['gen_lr'], 'weight_decay': args['weight_decay']}
    ])
    GR_optimizer = optim.Adam([
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'gen_lr': 2 * args['gen_lr']},
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'gen_lr': args['gen_lr'], 'weight_decay': args['weight_decay']}
    ])
    DS_optimizer = optim.Adam([
        {'params': [param for name, param in ds_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'dis_lr': 2 * args['dis_lr']},
        {'params': [param for name, param in ds_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'dis_lr': args['dis_lr'], 'weight_decay': args['weight_decay']}
    ])
    DR_optimizer = optim.Adam([
        {'params': [param for name, param in ds_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'dis_lr': 2 * args['dis_lr']},
        {'params': [param for name, param in ds_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'dis_lr': args['dis_lr'], 'weight_decay': args['weight_decay']}
    ])
    ###
    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        gs_net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        GS_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        GS_optimizer.param_groups[0]['gen_lr'] = 2 * args['gen_lr']
        GS_optimizer.param_groups[1]['gen_lr'] = args['gen_lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    curr_iter = args['last_iter']
    while True:
        train_loss_record = AvgMeter()
        train_L1loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()
        train_adv_loss_record = AvgMeter()
        train_DSnet_loss_record = AvgMeter()

        train_DCloss_record = AvgMeter()
        train_TVloss_record = AvgMeter()



        for i, data in enumerate(train_loader):
            GS_optimizer.param_groups[0]['gen_lr'] = 2 * args['gen_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            GS_optimizer.param_groups[1]['gen_lr'] = args['gen_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            GR_optimizer.param_groups[0]['gen_lr'] = 2 * args['gen_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                           ) ** args['lr_decay']
            GR_optimizer.param_groups[1]['gen_lr'] = args['gen_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                       ) ** args['lr_decay']
            DS_optimizer.param_groups[0]['dis_lr'] = 2 * args['dis_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                           ) ** args['lr_decay']
            DS_optimizer.param_groups[1]['dis_lr'] = args['dis_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                       ) ** args['lr_decay']
            DR_optimizer.param_groups[0]['dis_lr'] = 2 * args['dis_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                           ) ** args['lr_decay']
            DR_optimizer.param_groups[1]['dis_lr'] = args['dis_lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                       ) ** args['lr_decay']
            inputs, gts, dps,reals = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()
            reals = Variable(reals).cuda()
            GS_optimizer.zero_grad()
            GR_optimizer.zero_grad()
            # DS_optimizer.zero_grad()
            # DR_optimizer.zero_grad()
            ###
            GS_result, depth_pred = gs_net(inputs)
            syn_out=ds_net(GS_result)
            syn_out2=ds_net(gts)
            # syn_out, syn_out2 = ds_net(GS_result, gts)
            ###
            re_result, re_depth_pred = gs_net(reals)
            con_re,con_depth = gr_net(re_result)
            real_out = dr_net(re_result)
            real_out2 = dr_net(gts)
            # real_out, real_out2 = dr_net(re_result, gts)
            ####GANloss
            target_real = torch.ones(syn_out2.size()).float().cuda()  # 全填充为1
            target_fake = torch.zeros(syn_out.size()).float().cuda()  # 全填充为0
            MSE_criterion = nn.MSELoss()
            GS_advloss = 0.5 * MSE_criterion(syn_out, target_real)+0.5 * MSE_criterion(real_out, target_real)
            ###unsepervised_loss
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            re_result= re_result.to(device)
            TV_loss = TVLossL1(re_result)
            DC_loss = DCLoss(re_result, 35)
            CYC_loss = criterion(con_re, inputs)
            ###unsepervised_loss
            ###sepervised_loss
            l1loss = nn.L1Loss()
            loss_net = l1loss(GS_result, gts)
            loss_depth = l1loss(depth_pred, dps)
            loss = loss_net + loss_depth + 0.1*TV_loss+ 0.5*DC_loss+ 0.5*CYC_loss + 0.5*GS_advloss
            loss.backward()
            GS_optimizer.step()
            GR_optimizer.step()

            ###DiscriminativeNet_loss
            DS_optimizer.zero_grad()
            result_syn = result_buffer.push_and_pop(GS_result)
            gts = result_buffer.push_and_pop(gts)
            syn_out = ds_net(result_syn)
            syn_out2 = ds_net(gts)
            DS_loss = 0.5 * MSE_criterion(syn_out, target_fake) + 0.5 * MSE_criterion(syn_out2, target_real)
            DS_loss.backward()
            DS_optimizer.step()
            ####
            DR_optimizer.zero_grad()
            result_real = result_buffer.push_and_pop(re_result)
            gts = result_buffer.push_and_pop(gts)
            real_out = dr_net(result_real)
            real_out2 = dr_net(gts)
            DS_loss2 = 0.5 * MSE_criterion(real_out, target_fake) + 0.5 * MSE_criterion(real_out2, target_real)
            DS_loss2.backward()
            DR_optimizer.step()
            ###DiscriminativeNet_loss
            ###
            train_loss_record.update(loss.data, batch_size)
            train_L1loss_record.update(loss_net.data, batch_size)
            train_depth_loss_record.update(loss_depth.data, batch_size)
            train_adv_loss_record.update(GS_advloss.data, batch_size)
            train_DSnet_loss_record.update(DS_loss.data,batch_size)
            train_DCloss_record.update(DC_loss,batch_size)
            train_TVloss_record.update(TV_loss,batch_size)


            curr_iter += 1

            log = '[iter %d], [Total_loss %.13f], [gen_lr %.13f], [dis_lr %.13f],[L1_loss %.13f], [loss_depth %.13f], [loss_adv %.13f], [loss_ds %.13f], [loss_dc %.13f], [loss_tv %.13f]' % \
                  (curr_iter, train_loss_record.avg, GS_optimizer.param_groups[1]['gen_lr'],DS_optimizer.param_groups[1]['dis_lr'],
                   train_L1loss_record.avg, train_depth_loss_record.avg,train_adv_loss_record.avg,train_DSnet_loss_record.avg,train_DCloss_record.avg,train_TVloss_record.avg)
            # print(111)
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