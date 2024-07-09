import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict
faulthandler.enable()
import utils
from seq_scripts import seq_train, seq_eval, seq_feature_generation



class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)  # 设置随机种子
        self.device = utils.GpuDataParallel()  # 设置GPU
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)  # 打印的设置
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()   #从指定路径加载手语字典，这个字典可能包含手语识别任务中类别标签和对应的索引关系
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()  # 模型和优化器

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))  # 打印参数列表(baseline.yaml)
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):  # 每个epoch
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if eval_model:
                    test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                       'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("test WER: {}%".format(np.mean(test_wer)))
                    # self.recoder.print_log("test WER: {:05.2f}%".format(test_lstm_wer))
                    # self.recoder.print_log("test WER: {:05.2f}%".format(merge_test_wer))
                if save_model:
                    model_path = "{}test_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, np.mean(test_wer), epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)

        elif self.arg.phase == 'test':  # 测试模式
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # _, train_wer, _ = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.gloss_dict, self.arg.evaluate_tool)

            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')


        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):  # 保存配置参数
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):   # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)



    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        # 加载优化器
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)  # 创建优化器对象

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)   # 加载权重参数
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)  # 把模型放到GPU上
        print("Loading model finished.")
        self.load_data()  # 加载数据
        return model, optimizer    # 返回加载的模型和优化器对象： model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)  # 将模型的所有参数和缓冲区移动到指定的设备上
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)  # 使用torch.load函数加载指定路径weight_path的权重文件，并将其存储在state_dict中
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):  # 加载数据集和构建数据加载器
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        # 合并列表mode和train_flag
        dataset_list = zip(["train", "train_eval", "test"], [True, False, False])  # 布尔值为True，表示该数据集用于训练模型
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']   # 数据集的根目录
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag  # 指示数据加载器是否进行数据转换（例如数据增强）
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,  # 数据是否打乱
            drop_last=train_flag,   # 当剩余数据不够组成一个batch时，是否扔弃
            num_workers=self.arg.num_worker,  # if train_flag else 0      # 决定有几个进程来处理data loading
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):  # 动态导入指定名称的Python类
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:      # 加载phoenix14.yaml
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    processor = Processor(args)  # 对Processor对象进行初始化
    utils.pack_code("./", args.work_dir)
    processor.start()
