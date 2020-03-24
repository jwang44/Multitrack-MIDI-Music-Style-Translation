import argparse
import os
import tensorflow as tf
from My_model import cyclegan

parser = argparse.ArgumentParser()
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

#parser.add_argument('--train_data_dir', dest='train_data_dir', default='./data/train')
#parser.add_argument('--test_data_dir', dest='test_data_dir', default='./data/test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='train_batch_size')
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='the weight of cycle loss over the whole generator loss')
parser.add_argument('--sigma_d', dest='sigma_d', type=float, default=1.0, help='standard deviation for gaussian noise')
args = parser.parse_args()

def main(_):
    checkpoint_dir = './checkpoint'
    log_dir = './log'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # 自动把不适合GPU运行的op 放到CPU运行
    tfconfig.gpu_options.allow_growth = True
    # 使程序在开始时逐步增长显存使用量 而不是直接占用全部

    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        else:
            raise NameError("the phase arg must be 'train' or 'test'")

if __name__ == '__main__':
    tf.app.run()
