import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from evaluation.slt_eval.txt_calculation import bleu, chrf, rouge, wer_list

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):  # 训练
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):  # 循环遍历数据集加载器(loader)中的每个批次数据
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        loss = model.criterion_calculation(ret_dict, label, label_lgt)   # 损失
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            continue
        optimizer.zero_grad()   # 优化器梯度清零
        loss.backward()    # 对损失进行反向传播
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'    # 每50个batch打印一个日志：损失
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
    optimizer.scheduler.step()    # 在一个epoch要结束时，调整学习率
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):  # 测试
    model.eval()
    total_sent = []   # 预测值
    total_info = []
    loss_value = []   # 每个batch的loss值
    total_conv_sent = []
    wer = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])  # 视频数据
        vid_lgt = device.data_to_device(data[1])  # 视频长度
        label = device.data_to_device(data[2])  # 标签数据
        label_lgt = device.data_to_device(data[3])  # 标签长度
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)   # 预测结果(词)
            loss = model.criterion_calculation(ret_dict, label, label_lgt)
            loss_value.append(loss.item())

        total_info += [file_name.split("/")[0] for file_name in data[-1]]   # 测试数据的标签

        all_gloss_label = data[-1]
        total_sent += ret_dict['recognized_sents']   # 预测结果（句子）
        total_conv_sent += ret_dict['conv_sents']
        total_merge_sent = ret_dict['merge_sents']  # 模型预测的句子
        gloss_hyp = []  # 模型预测的词汇序列

        for sample_idx, sample in enumerate(total_merge_sent):
            x = []
            for word_idx, word in enumerate(sample):
                x.append(word[0])
            gloss_hyp += [' '.join(x).upper()]
        gloss_ref = [''.join(t) for t in all_gloss_label]
        gloss_wer = wer_list(references=gloss_ref, hypotheses=gloss_hyp)
        wer.append(round(gloss_wer["wer"], 2))

    try:
        python_eval = True if evaluate_tool == "python" else False
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info, total_conv_sent)
        write2file(work_dir + "output-hypothesis-{}-merge.ctm".format(mode), total_info, total_merge_sent)

        conv_ret = evaluate(
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="result.txt",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=True,
        )
        merge_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-merge.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )

    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
        # recoder.print_log(f"Epoch {epoch}, {mode} {conv_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
        # recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
        # recoder.print_log(f"Epoch {epoch}, {mode} {merge_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
        recoder.print_log(f"Epoch {epoch}, {mode} {np.mean(wer): 2.2f}%", f"{work_dir}/{mode}.txt")
        recoder.print_log('\tMean Dev loss: {:.10f}.'.format(np.mean(loss_value)))
        # return conv_ret, lstm_ret, merge_ret
        return wer


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
    	os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
