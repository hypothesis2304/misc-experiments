import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as preprocess
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
import random
from test import image_classification_test
from tqdm import tqdm, trange
import copy
import random
from aug import *

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '42'


def train(config):
    base_network = network.ResNetFc('ResNet50', use_bottleneck=True, bottleneck_dim=config["bottleneck_dim"], new_cls=True)
    ad_net = network.AdversarialNetwork(config["bottleneck_dim"], config["hidden_dim"])

    base_network = base_network.cuda()
    ad_net = ad_net.cuda()

    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    source_path = ImageList(open(config["s_path"]).readlines(), transform=preprocess.image_train(resize_size=256, crop_size=224))
    target_path = ImageList(open(config["t_path"]).readlines(), transform=preprocess.image_train(resize_size=256, crop_size=224))
    test_path   = ImageList(open(config["t_path"]).readlines(), transform=preprocess.image_test(resize_size=256, crop_size=224))

    source_loader = DataLoader(source_path, batch_size=config["train_bs"], shuffle=True, num_workers=0, drop_last=True)
    target_loader = DataLoader(target_path, batch_size=config["train_bs"], shuffle=True, num_workers=0, drop_last=True)
    test_loader   = DataLoader(test_path, batch_size=config["test_bs"], shuffle=True, num_workers=0, drop_last=True)

    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config["gpus"].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])


    len_train_source = len(source_loader)
    len_train_target = len(target_loader)

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    best_model_path = None

    for i in trange(config["iterations"], leave=False):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(test_loader, base_network)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = copy.deepcopy(temp_model)
                best_iter = i
                if best_model_path and osp.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except:
                        pass
                best_model_path = osp.join(config["output_path"], "iter_{:05d}.pth.tar".format(best_iter))
                torch.save(best_model, best_model_path)
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(source_loader)
        if i % len_train_target == 0:
            iter_target = iter(target_loader)

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_cut, labels_cut = cutmix(base_network, inputs_source, labels_source, inputs_target, config["alpha"])
        inputs_mix, labels_mix = mixup(base_network, inputs_source, labels_source, inputs_target, config["alpha"])
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config["method"] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    print("Training Finished! Best Accuracy: ", best_acc)
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN', 'DANN+E'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/webcam_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/amazon_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--num_iterations', type=int, default=100004, help="number of iteration model should run")
    parser.add_argument('--test_size', type=int, default=4, help="batch_size for testing data")
    parser.add_argument('--alpha', type=int, default=1, help="for mixup and cutmix")
    parser.add_argument('--batch_size', type=int, default=12, help="batch_size for training data")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="size of bottleneck after feature_extractor")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="size of hidden layer after feature_extractor")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    config = {}
    config["dset"] = args.dset
    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    if config["dset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003
        class_num = 31
    elif config["dset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001
        class_num = 65


    # train config
    config["train_bs"] = args.batch_size
    config["test_bs"] = args.test_size
    config["iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot"] = args.snapshot_interval
    config["alpha"] = args.alpha
    config["s_path"] = args.s_dset_path
    config["t_path"] = args.t_dset_path
    config["gpus"] = args.gpu_id
    config["method"] = args.method
    config["bottleneck_dim"] = args.bottleneck_dim
    config["hidden_dim"] = args.hidden_dim
    config["output_path"] = "experiments/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system("mkdir -p " + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    param_file = open(osp.join(config["output_path"], "params.txt"), "w")

    param_file.write("test_interval: " + str(args.test_interval) + "\n")
    param_file.write("method: " + args.method + "\n")
    param_file.write("s_dset_path: "+ args.s_dset_path+ "\n")
    param_file.write("t_dset_path: "+ args.t_dset_path+ "\n")
    param_file.write("lr: "+ str(args.lr) + "\n")
    param_file.write("alpha: "+ str(args.alpha)+ "\n")
    param_file.write("test_interval: "+ str(args.test_interval)+ "\n")
    param_file.write("bottleneck_dim: "+ str(args.bottleneck_dim)+ "\n")
    param_file.write("hidden_dim: "+ str(args.hidden_dim)+ "\n")
    param_file.flush()
    train(config)
