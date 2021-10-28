import argparse
import data_utils
import models
import tensorflow.compat.v1 as tf
import numpy as np
import os

from tqdm import tqdm
from optimizers.optimizer import Optimizer
from losses.loss import Loss
from tools import print_net_info, print_args, print_training_info, Timer

tf.logging.set_verbosity(tf.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    # ===================数据集相关参数=====================
    # 使用的数据集
    parser.add_argument('--dataset', type=str, default='mnist')
    # 数据路径
    parser.add_argument('--data_path', type=str, required=True, nargs='+')
    # 训练使用的batch_size
    parser.add_argument('--batch_size', type=int, default=128)
    # 是否需要对图像进行归一化
    parser.add_argument('--not_normalize', 
                        action='store_true', default=False)
    # 取数据时是否需要以随机方式读取
    parser.add_argument('--not_shuffle', 
                        action='store_true', default=False)
    # 是否是Cifar-10数据集
    parser.add_argument('--c10', action='store_true', default=False)
    # 是否使用Cifar-100的粗略标签（仅对Cifar100有效）
    parser.add_argument('--coarse_label', 
                        action='store_true', default=False)
    # 训练集与测试集的比例
    parser.add_argument('--split_train_and_test', type=float, default=0.2)
    # 是否需要对输入图像缩放（对非统一尺寸的Oxford Flower数据集有效）
    parser.add_argument('--resize', type=int, nargs='+')
    # 输入图像是否属于小图像
    parser.add_argument('--not_small', action='store_true', default=False)
    # 是否使用数据增强
    parser.add_argument('--augmentation', 
                        action='store_true', default=False)
    # =====================================================
    # =================模型相关参数=========================
    # 使用的模型
    parser.add_argument('--model', type=str)
    # 模型的结构（深度等）
    parser.add_argument('--structure', type=int)
    # =====================================================
    # ===================训练相关参数=======================
    # 训练周期数
    parser.add_argument('--epoch', type=int, default=350)
    # 训练使用的优化器
    parser.add_argument('--optim', type=str, default='adam')
    # 训练使用的损失函数
    parser.add_argument('--loss', type=str, default='mse', 
                        choices=['mse', 'ce'])
    # 训练使用的学习率
    parser.add_argument('--lr', type=float, default=0.01)
    # 学习率变化递减的周期数
    parser.add_argument('--boundary', type=int, 
                        nargs='*', default=[160, 250])
    # 学习率递减倍数
    parser.add_argument('--decay', type=float, default=0.1)
    # 是否对学习率使用warmup
    parser.add_argument('--warmup', action='store_true', default=False)
    # warmup的周期数
    parser.add_argument('--warmup_epoch', type=int, default=5)
    # =====================================================
    # ===============权值与日志相关参数======================
    # 是否存储权重文件
    parser.add_argument('--not_save_ckpt', 
                        action='store_true', default=False)
    # 权重文件路径
    parser.add_argument('--ckpt_path', type=str, default='checkpoint')
    # 日志路径
    parser.add_argument('--log_dir', type=str, default='logs')
    # =====================================================
    # =====================================================
    # 是否是测试阶段（训练/测试阶段）
    parser.add_argument('--testing', action='store_true', default=False)
    # =====================================================

    args = parser.parse_args()
    
    exp_name = '{}{}_{}'.format(args.model, args.structure, args.dataset)
    args.ckpt_path = os.path.join(args.ckpt_path, exp_name)
    args.log_dir = os.path.join(args.log_dir, exp_name)

    return args


def main(args):
    # 打印命令行传入的参数
    print_args(args)

    # 得到特定数据集的实例
    data = data_utils.get_dataloader(args)

    # 得到特定数据集的样本与标签的占位符
    X, Y = data_utils.get_placeholders(args)

    # 由于网络中使用训练与测试阶段表现不同的BN,需要指定当前阶段
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # 根据用户传入的参数值获取模型
    network_builder = models.get_model_with_name(args.model)
    network = network_builder(args.structure, 
                              class_num=Y.get_shape().as_list()[-1], 
                              is_small=not args.not_small)
    
    # 为模型传入输入张量，得到模型输出
    pred = network.build(X, is_training)

    with Timer('{}\n{}'.format('=' * 100, 
                               'Total {} time'.format('testing' 
                                                      if args.testing 
                                                      else 'training'))):
        with tf.Session() as sess:
            if args.testing:
                # 若用户指定为测试阶段
                acc = test(X, Y, is_training, pred, data, args, sess)
                print('Test acc: {}'.format(acc))
            else:
                # 用户指定为训练阶段
                train(X, Y, is_training, pred, data, args, sess)


def train(X, Y, is_training, pred, data, args, sess):
    # 写入模型训练的计算图与各种参数信息
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    # 打印模型的信息（参数量与参数信息）
    print_net_info()

    # 一个epoch中含有的迭代数
    iter_num = int(data.num_examples('train') / args.batch_size)

    # 根据用户传入的损失函数类型得到损失值
    cost = Loss(args.loss).get_loss(Y, pred)
    # 将损失值写入日志
    tf.summary.scalar('loss', cost)

    # 根据用户传入的训练相关参数初始化优化器并对损失值进行优化
    optim_op = Optimizer(initial_lr=args.lr, 
                         boundary=[iter_num * b for b in args.boundary], 
                         decay=args.decay, 
                         warmup=args.warmup,
                         warmup_iter=args.warmup_epoch * iter_num,
                         name=args.optim).minimize(cost)

    saver = None
    # 如果用户没有传入已有的权值路径，则创建
    if not args.not_save_ckpt:
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        
        saver = tf.train.Saver()

    # 记录当前最好的准确率和损失值
    best_acc = 0
    best_loss = np.Inf
    best_epoch = 0

    merged_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    # 训练用户指定的epoch数
    for e in range(args.epoch):
        epoch_loss = list()

        # 每一个epoch内都需要训练iter_num次
        for _ in tqdm(range(iter_num)):
            # 取出一个batch数据
            x, y = data.next_batch('train')
            # 计算当前迭代的损失以及优化过程
            iter_cost, _ = sess.run([cost, optim_op], 
                                     feed_dict={X: x, Y: y, is_training: True})
            # 将当前迭代的损失加入当前周期的损失
            epoch_loss.append(iter_cost)

        # 每训练完成一个周期都进行一次测试，观察模型在测试集的表现
        acc = test(X, Y, is_training, pred, data, args, sess)
        
        # 添加日志信息
        summ = sess.run(merged_op, feed_dict={X: x, Y: y, is_training: True})
        writer.add_summary(summ, e)

        # 计算周期内的平均损失
        epoch_loss = np.mean(epoch_loss)
        # 打印训练信息
        print_training_info(e, epoch_loss, acc)
        
        # 记录最佳准确率与损失值
        if acc > best_acc:
            best_acc = acc
            best_loss = epoch_loss
            best_epoch = e
            if saver:
                saver.save(sess, 
                           os.path.join(args.ckpt_path, 'checkpoint'), 
                           global_step=e)
                print('Saving...')
    # 打印模型表现最好的情况
    print('{}\n{}'.format('=' * 100, 'Best: '))
    print_training_info(best_epoch, best_loss, best_acc)


def test(X, Y, is_training, pred, data, args, sess):
    # 计算模型预测结果的最大分量位置是否与标签最大分量位置相同
    # 若相同，则表示当前模型对于输入样本预测正确
    # 以此统计模型预测正确数量
    correct_num = tf.reduce_sum(tf.cast(
                        tf.equal(
                            tf.math.argmax(pred, axis=-1), 
                            tf.math.argmax(Y, axis=-1)
                        ), tf.float32)
                  )

    correct = 0
    total = 0

    # 如果当前是单独的测试阶段，则从传入的路径中读取最新的权值文件信息
    if args.testing:
        saver = tf.train.Saver()
        latest_ckpt = tf.train.latest_checkpoint(args.ckpt_path)
        saver.restore(sess, latest_ckpt)

    # 测试集需要的迭代数
    iter_num = int(data.num_examples('test') / args.batch_size)
    
    for _ in tqdm(range(iter_num)):
        x, y = data.next_batch('test')
        # 对每一个batch的数据统计一次预测正确的数目
        # 将当前batch内预测正确的数目累计在正确总数上
        correct += sess.run(correct_num, 
                            feed_dict={X: x, Y: y, is_training: False})
        # 使用total变量记录所有输入样本数量
        total += args.batch_size

    # 使用预测正确的数据/总输入数量，即为测试集上的准确率
    return correct / total


if __name__ == "__main__":
    args = parse_args()
    main(args)
