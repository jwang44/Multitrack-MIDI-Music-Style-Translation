import tensorflow as tf
# from module import *
import numpy as np
import os
from module import conv2d, relu, lrelu, instance_norm, deconv2d, load_npy_data
import datetime, time
import ops
import glob

# import collections

class cyclegan():
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.lr = args.lr
        self.beta1 = args.beta1
        self.L1_lambda = args.L1_lambda # cycle_loss 在生成器误差中的权重
        self.sigma_d = args.sigma_d
        self.time_step = 64 # 量化到16分音符，每个样本4小节
        self.pitch_range = 84
        # self.generator = generator_resnet ##待定
        #OPTIONS = collections.namedtuple()
        self.build_model()
        self.saver = tf.train.Saver()


    def build_model(self):
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range, 2])
        self.real_A = self.real_data[:, :, :, :1]
        self.real_B = self.real_data[:, :, :, 1:2]
        self.gaussian_noise = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range, 1])


        def generator(image, reuse= False, name="generator"):
            with tf.variable_scope(name):

                if reuse:
                    tf.get_variable_scope().reuse_variables()

                def residule_block(x, dim, ks=3, s=1, name='res'):
                    # e.g, x is (# of images * 128 * 128 * 3)
                    p = int((ks - 1) / 2)
                    # For ks = 3, p = 1
                    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                    # After first padding, (# of images * 130 * 130 * 3)
                    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
                    # After first conv2d, (# of images * 128 * 128 * 3)
                    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                    # After second padding, (# of images * 130 * 130 * 3)
                    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
                    # After second conv2d, (# of images * 128 * 128 * 3)
                    return relu(y + x)

                # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
                # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
                # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3

                # Original image is (# of images * 256 * 256 * 3)
                c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                # c0 is (# of images * 262 * 262 * 3)
                c1 = relu(instance_norm(conv2d(c0, 64, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
                # c1 is (# of images * 256 * 256 * 64)
                c2 = relu(instance_norm(conv2d(c1, 64 * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
                # c2 is (# of images * 128 * 128 * 128)
                c3 = relu(instance_norm(conv2d(c2, 64 * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
                # c3 is (# of images * 64 * 64 * 256)

                # c4 = relu(instance_norm(conv2d(c3, options.gf_dim*8, 3, 3, name='g_e4_c'), 'g_e4_bn'))
                # c5 = relu(instance_norm(conv2d(c4, options.gf_dim*16, 3, [4, 1], name='g_e5_c'), 'g_e5_bn'))

                # define G network with 9 resnet blocks
                r1 = residule_block(c3, 64 *4, name='g_r1')
                # r1 is (# of images * 64 * 64 * 256)
                r2 = residule_block(r1, 64*4, name='g_r2')
                # r2 is (# of images * 64 * 64 * 256)
                r3 = residule_block(r2, 64*4, name='g_r3')
                # r3 is (# of images * 64 * 64 * 256)
                r4 = residule_block(r3, 64*4, name='g_r4')
                # r4 is (# of images * 64 * 64 * 256)
                r5 = residule_block(r4, 64*4, name='g_r5')
                # r5 is (# of images * 64 * 64 * 256)
                r6 = residule_block(r5, 64*4, name='g_r6')
                # r6 is (# of images * 64 * 64 * 256)
                r7 = residule_block(r6, 64*4, name='g_r7')
                # r7 is (# of images * 64 * 64 * 256)
                r8 = residule_block(r7, 64*4, name='g_r8')
                # r8 is (# of images * 64 * 64 * 256)
                r9 = residule_block(r8, 64*4, name='g_r9')
                # r9 is (# of images * 64 * 64 * 256)
                r10 = residule_block(r9, 64*4, name='g_r10')

                # d4 = relu(instance_norm(deconv2d(r9, options.gf_dim*8, 3, [4, 1], name='g_d4_dc'), 'g_d4_bn'))
                # d5 = relu(instance_norm(deconv2d(d4, options.gf_dim*4, 3, 3, name='g_d5_dc'), 'g_d5_bn'))

                d1 = relu(instance_norm(deconv2d(r10, 64 * 2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
                # d1 is (# of images * 128 * 128 * 128)
                d2 = relu(instance_norm(deconv2d(d1, 64, 3, 2, name='g_d2_dc'), 'g_d2_bn'))
                # d2 is (# of images * 256 * 256 * 64)
                d3 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                # After padding, (# of images * 262 * 262 * 64)
                pred = tf.nn.sigmoid(conv2d(d3, 1, 7, 1, padding='VALID', name='g_pred_c'))
                # Output image is (# of images * 256 * 256 * 3)

                return pred 

        def discriminator(image, reuse=False, name="discriminator"):
            with tf.variable_scope(name):

                if reuse:
                    tf.get_variable_scope().reuse_variables()

                h0 = lrelu(conv2d(image, output_dim=64, name="d_h0_conv"))
                h1 = lrelu(instance_norm(conv2d(h0, output_dim=64*4, name="d_h1_conv")))
                h4 = conv2d(h1, 1, s=1, name="d_h3_pred")
                return h4

        self.fake_A = generator(self.real_B,  name="generatorB2A")
        self.fake_B_ = generator(self.fake_A, name="generatorA2B")

        self.fake_B = generator(self.real_A, reuse=True, name="generatorA2B")
        self.fake_A_ = generator(self.fake_B, reuse=True, name="generatorB2A")

        # velocity 二值化
        self.real_A_binary = ops.to_binary(self.real_A, threshold=0.5)
        self.real_B_binary = ops.to_binary(self.real_B, threshold=0.5)
        self.fake_A_binary = ops.to_binary(self.fake_A, threshold=0.5)
        self.fake_B_binary = ops.to_binary(self.fake_B, threshold=0.5)
        self.fake_A__binary = ops.to_binary(self.fake_A_, threshold=0.5)
        self.fake_B__binary = ops.to_binary(self.fake_B_, threshold=0.5)
        
        self.DA_real = discriminator(self.real_A + self.gaussian_noise, name="discriminatorA")
        self.DA_fake = discriminator(self.fake_A + self.gaussian_noise, reuse=True, name="discriminatorA")
        self.DB_real = discriminator(self.real_B + self.gaussian_noise, name="discriminatorB")
        self.DB_fake = discriminator(self.fake_B + self.gaussian_noise, reuse=True, name="discriminatorB")

        # cycle consistency loss
        self.cycle_loss = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A - self.fake_A_)) + \
            self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
        
        # adversarial loss
        self.g_loss_a2b = tf.reduce_mean((self.DB_fake - tf.ones_like(self.DB_fake))**2)
        self.g_loss_b2a = tf.reduce_mean((self.DA_fake - tf.ones_like(self.DA_fake))**2)
        # 生成器总误差
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a + self.cycle_loss

        # 判别器误差
        self.d_loss_A_real = tf.reduce_mean((self.DA_real - tf.ones_like(self.DA_real))**2)
        self.d_loss_B_real = tf.reduce_mean((self.DB_real - tf.ones_like(self.DB_real))**2)
        self.d_loss_A_fake = tf.reduce_mean((self.DA_fake - tf.zeros_like(self.DA_fake))**2)
        self.d_loss_B_fake = tf.reduce_mean((self.DB_real - tf.zeros_like(self.DB_fake))**2)
        
        self.d_loss_A = (self.d_loss_A_fake + self.d_loss_A_real) / 2
        self.d_loss_B = (self.d_loss_B_fake + self.d_loss_B_real) / 2
        self.d_loss = self.d_loss_A + self.d_loss_B

        # 定义所有 summary，用于 tensorboard 可视化
        self.cycle_loss_sum = tf.summary.scalar('cycle_loss', self.cycle_loss)
        self.g_loss_a2b_sum = tf.summary.scalar('g_loss_a2b', self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar('g_loss_b2a', self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.cycle_loss_sum, self.g_loss_sum])
        
        self.d_loss_A_sum = tf.summary.scalar('d_loss_A', self.d_loss_A)
        self.d_loss_B_sum = tf.summary.scalar('d_loss_B', self.d_loss_B)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        self.d_sum = tf.summary.merge([self.d_loss_A_sum, self.d_loss_B_sum, self.d_loss_sum])



        # 测试样本 placeholder
        self.test_A = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range, 1], name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range, 1], name='test_B')
        
        self.testB = generator(self.test_A, reuse=True, name="generatorA2B")
        self.testA_ = generator(self.testB, reuse=True, name="generatorB2A")
        
        self.testA = generator(self.test_B, reuse=True, name="generatorB2A")
        self.testB_ = generator(self.testA, reuse=True, name="generatorA2B")

        # to binary
        self.test_A_binary = ops.to_binary(self.test_A, 0.5)
        self.test_B_binary = ops.to_binary(self.test_B, 0.5)
        self.testB_binary = ops.to_binary(self.testB, 0.5)
        self.testA__binary = ops.to_binary(self.testA_, 0.5)
        self.testA_binary = ops.to_binary(self.testA, 0.5)
        self.testB__binary = ops.to_binary(self.testB_, 0.5)

        trainable_vars = tf.trainable_variables()
        self.d_vars = []
        self.g_vars = []
        for var in trainable_vars:
            if 'discriminator' in var.name:
                self.d_vars.append(var)
            if 'generator' in var.name:
                self.g_vars.append(var)
            print(var.name)

    def train(self, args):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optimizer = tf.train.AdamOptimizer(self.lr, self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self.lr, self.beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op) # 初始化全局变量

        log_dir = './log/{}'.format(datetime.datetime.now())
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # load_data
        data_A = glob.glob("./data/JC_C/train/*.*")
        data_B = glob.glob("./data/JC_J/train/*.*")

        start_time = time.time()
        for epoch in range(self.epoch):
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)

            batches = min(len(data_A), len(data_B)) // self.batch_size
            # 暂时略去 learning rate 衰减
            lr = args.lr

            for batch in range(batches):
                batch_files = zip(
                    data_A[batch*self.batch_size:(batch+1)*self.batch_size],
                    data_B[batch*self.batch_size:(batch+1)*self.batch_size]
                    ) # 把 dataA dataB 两个列表中元素 一一对应地组成元组
                batch_files = list(batch_files) # 转换为列表，列表中的元素是元组
                batch_images = [load_npy_data(batchfile) for batchfile in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                gaussian_noise = np.abs(np.random.normal(0, self.sigma_d, [self.batch_size, self.time_step, self.pitch_range, 1]))

                fake_A, fake_B, _, summary_str, g_loss_a2b, g_loss_b2a, cycle_loss, g_loss = \
                    self.sess.run([self.fake_A, self.fake_B, self.g_optimizer, 
                    self.g_sum, self.g_loss_a2b, self.g_loss_b2a, self.cycle_loss, self.g_loss],
                    feed_dict={self.real_data: batch_images, self.gaussian_noise: gaussian_noise, self.lr: lr})
                    # 是否需要 feed_dict 取决于前面那些变量的产生是否需要用到 placeholder

                _, summary_str, da_loss, db_loss, d_loss = self.sess.run([
                    self.d_optimizer, self.d_sum, self.d_loss_A, self.d_loss_B, self.d_loss],
                    feed_dict = {self.real_data: batch_images, self.gaussian_noise: gaussian_noise, 
                                self.lr: lr} )
                
                print('=================================================================')
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %6.2f, G_loss: %6.2f" %
                      (epoch, batch, batches, time.time() - start_time, d_loss, g_loss)))
                print(("++++++++++G_loss_A2B: %6.2f G_loss_B2A: %6.2f Cycle_loss: %6.2f DA_loss: %6.2f DB_loss: %6.2f" %
                       (g_loss_a2b, g_loss_b2a, cycle_loss, da_loss, db_loss)))

            self.save(args)  


    def save(self, args):
        model_dir = "try_saving_{}".format(datetime.datetime.now())
        checkpoint_dir = os.path.join(args.checkpoint_dir, model_dir)
        self.saver.save(self.sess, save_path= checkpoint_dir)
    
    def load(self, args):
        print("-----------READING CHECKPOINT-------------")
        checkpoint_dir = "./checkpoint/{}".format(args.date)
        pass

    def test(self,args):
        pass