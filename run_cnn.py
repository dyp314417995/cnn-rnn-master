#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 字典
def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    # 判断文件存放的路径是否存在，不存在则创建
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    # loss ：损失函数，交叉熵
    # acc：准确率
    # tf.summary.scalar:用来显示标量信息
    # 标量：只有数值大小，没有方向，也叫“无向量”
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    # 将所有summary全部保存到磁盘，以便 tensorboard 显示
    merged_summary = tf.summary.merge_all()
    # 指定一个文件用来保存图。
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    # 保存训练结果的对象
    # save_dir：保存结果的路径:'checkpoints/textcnn'
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    # 开始时间为当前时间
    start_time = time.time()
    # train_dir：训练集的文件：cnews.train.txt
    # val_dir：验证集的文件：cnews.val.txt
    # word_to_id：词汇表对应的字典
    # cat_to_id：目录对应的字典
    # config.seq_length：序列长度:600
    # process_file：将文件转换为id表示
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    print("x_train" + x_train)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    """获取已使用时间"""
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    # 全局变量初始化
    session.run(tf.global_variables_initializer())
    # 添加图？不是很懂
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    # num_epochs = 10  # 总迭代轮次
    for epoch in range(config.num_epochs):
        # 因为epoch从0开始
        print('Epoch:', epoch + 1)
        # batch_size = 64  # 每批训练大小
        """batch_iter:生成批次数据"""
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            # dropout_keep_prob = 0.5  # dropout保留比例
            # feed_data：将数据转换成字典
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            # save_per_batch = 10  # 每多少轮存入tensorboard
            # % 取模，即取余，也就是save_per_batch被total_batch整除时候写入tensorboard scalar
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                # 原来的写法s = session.run(merged_summary, feed_dict=feed_dict)
                # 可以用以下来代替
                s = session.run(merged_summary, feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo
                # 如果本次计算的结果比之前记录的最佳结果更好，就要更新
                # last_improved:记录上一次提升批次
                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print("msg.format" + msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,
                                                improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1
            # require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")
    print('Configuring CNN model...')
    config = TCNNConfig()
    # 判断文件是否存在，参数为路径带文件名
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        # 重建词汇表，参数3个：训练路径，词汇表路径，词汇表达小
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    # 这是什么操作？修改词汇表的大小
    config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
