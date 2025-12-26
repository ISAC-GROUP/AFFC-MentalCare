from utils import *
import copy
import torch.nn as nn
import numpy as np
import os.path as osp

CUDA = torch.cuda.is_available()


def train_one_epoch(args, multi_loss, data_loader, net, loss_fn, optimizer, alpha):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    if multi_loss:
        for i, (x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, domain_y_batch) in enumerate(data_loader):
            if CUDA:
                x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, domain_y_batch = x0_batch.cuda(), x1_batch.cuda(), x2_batch.cuda(), x3_batch.cuda(), x4_batch.cuda(), y_batch.cuda(), domain_y_batch.cuda()
            if args.model in args.domain_model:
                in_loss, out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, domain_y_batch, alpha)
                loss = loss_fn(out, y_batch) + sum(in_loss)
                #print('in_loss: ', [loss.item() for loss in in_loss], sum(in_loss))
            else:
                in_loss, out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, domain_y_batch, alpha)
                loss = loss_fn(out, y_batch) + in_loss
            _, pred = torch.max(out, 1)
            pred_train.extend(pred.data.tolist())
            act_train.extend(y_batch.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl.add(loss.item())
    else:
        for i, (x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, _) in enumerate(data_loader):  
            if CUDA:
                x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch = x0_batch.cuda(), x1_batch.cuda(), x2_batch.cuda(), x3_batch.cuda(), x4_batch.cuda(), y_batch.cuda()
            class_out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch) 
            loss = loss_fn(class_out, y_batch)
            _, pred = torch.max(class_out, 1)  
            pred_train.extend(pred.data.tolist())
            act_train.extend(y_batch.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(args, multi_loss, data_loader, net, loss_fn, alpha):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    if multi_loss:
        with torch.no_grad():  
            for i, (x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, domain_y_batch) in enumerate(data_loader):
                if CUDA:
                    x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, domain_y_batch = x0_batch.cuda(), x1_batch.cuda(), x2_batch.cuda(), x3_batch.cuda(), x4_batch.cuda(), y_batch.cuda(), domain_y_batch.cuda()
                if args.model in args.domain_model:
                    if args.model in ['CDPT_simple', 'CDPT_simple_wo_cl', 'MDNet_wo_modality', 'MDNet_wo_mod_w_crossatt','MDNet_wo_mod_w_cross_cl',
                    'New_CDPT_wo_gsr','New_CDPT_wo_ppg','New_CDPT_wo_ppg_ir','New_CDPT_wo_ir_gsr','New_CDPT_wo_ppg_gsr','New_CDPT_wo_ir_skt',
                    'New_CDPT_wo_ppg_ir_ired','New_CDPT_wo_ppg_ir_skt','New_CDPT_wo_ppg_ired_skt','New_CDPT_wo_ir_ired_skt'] and args.test_mode:
                        out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, domain_y_batch, alpha)
                        print(out, type(out))
                    else:
                        in_loss, out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, domain_y_batch, alpha)
                else:
                    in_loss, out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch)
                if args.model in ['CDPT_simple', 'CDPT_simple_wo_cl', 'MDNet_wo_modality', 'MDNet_wo_mod_w_crossatt','MDNet_wo_mod_w_cross_cl',
                'New_CDPT_wo_gsr','New_CDPT_wo_ppg','New_CDPT_wo_ppg_ir','New_CDPT_wo_ir_gsr','New_CDPT_wo_ppg_gsr','New_CDPT_wo_ir_skt',
                'New_CDPT_wo_ppg_ir_ired','New_CDPT_wo_ppg_ir_skt','New_CDPT_wo_ppg_ired_skt','New_CDPT_wo_ir_ired_skt']:
                    loss = loss_fn(out, y_batch)
                else:
                    loss = loss_fn(out, y_batch) + sum(in_loss)
                _, pred = torch.max(out, 1)
                vl.add(loss.item())
                pred_val.extend(pred.data.tolist())
                act_val.extend(y_batch.data.tolist())
    else:
        with torch.no_grad():  
            for i, (x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch, _) in enumerate(data_loader):
                if CUDA:
                    x0_batch, x1_batch, x2_batch, x3_batch, x4_batch, y_batch = x0_batch.cuda(), x1_batch.cuda(), x2_batch.cuda(), x3_batch.cuda(), x4_batch.cuda(), y_batch.cuda()
                out = net(x0_batch, x1_batch, x2_batch, x3_batch, x4_batch)
                loss = loss_fn(out, y_batch)
                _, pred = torch.max(out, 1)
                vl.add(loss.item())
                pred_val.extend(pred.data.tolist())
                act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val  

def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, train_loader, val_loader, subject):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial'
    set_up(args)

    model = get_model(args)
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    def save_model(name):  # max-acc
        model_path = osp.join(args.save_path, 'Best_Models', args.model + '_' + args.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        previous_model = osp.join(model_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(model_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    patience = 30  
    no_improve_count = 0 
    best_epoch = 0 

    timer = Timer()
    current_step, total_steps = 1, args.max_epoch
    for epoch in range(1, args.max_epoch + 1):
        p = current_step / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        loss_train, pred_train, act_train = train_one_epoch(args,
            multi_loss=args.multi_loss, data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer, alpha=alpha)

        acc_train, precision_train, recall_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} pre={:.4f} rec={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, precision_train, recall_train, f1_train))

        loss_val, pred_val, act_val = predict(
            args, multi_loss=args.multi_loss, data_loader=val_loader, net=model, loss_fn=loss_fn, alpha=alpha
        )
        acc_val, prediction_val, recall_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} pre={:.4f} rec={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, prediction_val, recall_val, f1_val))

        if acc_val > trlog['max_acc']: 
            print('epoch {}, acc_val {} > max_acc {}, update model'.format(epoch, acc_val, trlog['max_acc']))
            trlog['max_acc'] = acc_val
            best_epoch = epoch
            no_improve_count = 0
            save_model(name=str(subject) + '-max-acc')  
        elif args.early_stop:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f'Early stopping at epoch {epoch} (best epoch: {best_epoch})')
                break
        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                        subject))
    save_name_ = args.model + '_trlog' + save_name
    ensure_path(osp.join(args.save_path, 'Output', 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'Output', 'log_train', save_name_))  
    return trlog['max_acc']


def test(args, test_loader, reproduce):
    seed_all(args.random_seed)
    set_up(args)
    subject = args.sub_id
    args.test_mode = True
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if args.reproduce:
        model_name_reproduce = '{}-max-acc.pth'.format(subject) 
        data_type = '{}_{}'.format(args.model, args.dataset)
        save_path = osp.join(args.save_path, data_type)
        ensure_path(save_path)
        model_name_reproduce = osp.join(save_path, model_name_reproduce)
        model.load_state_dict(torch.load(model_name_reproduce))
    else:
        best_model_path = args.load_path + 'Best_Models/' + args.model + '_' + args.dataset + '/' + str(subject) + '-max-acc.pth'
        model.load_state_dict(torch.load(best_model_path))
    loss, pred, act = predict(
        args=args, multi_loss=args.multi_loss, data_loader=test_loader, net=model, loss_fn=loss_fn, alpha=1)
    acc, _, _, f1, _ = get_metrics(y_pred=pred, y_true=act)
    return acc, pred, act


