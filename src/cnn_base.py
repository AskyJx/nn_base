import os
import struct
import numpy as np
import logging.config
import random, time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numba

# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/../config/logging.conf'
# log_conf = exec_abs + '\\config\\logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('main')

# 持久化配置
trace_file_path = 'D:/0tmp/'
exec_name = os.path.basename(__file__)
trace_file = trace_file_path + exec_name + ".data"

# 训练数据
path_minst_unpack = 'F:\cwork\Project\TF_py3.6\MNIST_data_unpack'

# General params
class Params:
    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True
    INIT_W = 0.01  # 权值初始化参数
    # 参考Le Cun Paper 设置learning rate
    # 增加了0次循环，并且FC和CONV1的学习率分别乘以特定系数
    DIC_L_RATE = {1: 0.001, 2: 0.0005, 3: 0.0002, 4: 0.0001, 5: 0.00005, 6: 0.00001, 7: 0.000005, 100: 0.000002}

    # Adam params
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    EPS2 = 1e-10
    REG_PARA = 0.5  # 正则化乘数
    LAMDA = 1e-4  # 正则化系数lamda
    EPOCH_NUM = 10  # EPOCH
    MINI_BATCH_SIZE = 200  # batch_size
    ITERATION = 1  # 每batch训练轮数
    TYPE_K = 10  # 分类类别
    DROPOUT_RATE = 0.5  # dropout%
    VALIDATION_CAPACITY = 2000  # 验证集大小
    VAL_FREQ = 50  # val per how many tIter
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1  # MINST通道数为，可支持多通道

    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32

    # Hyper params
    CONV1_F_SIZE = 5
    CONV1_STRIDES = 1
    CONV1_O_SIZE = 28
    CONV1_O_DEPTH = 32

    POOL1_F_SIZE = 2
    POOL1_STRIDES = 2

    CONV2_F_SIZE = 5
    CONV2_STRIDES = 1
    CONV2_O_SIZE = 14
    CONV2_O_DEPTH = 64

    POOL2_F_SIZE = 2
    POOL2_STRIDES = 2

    FC1_SIZE_INPUT = 3136
    FC1_SIZE_OUTPUT = 512

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)

# General Tools
class Tools:
    # padding before cross-correlation and pooling
    @staticmethod
    def padding(x, pad):

        size_x = x.shape[2]  # 输入矩阵尺寸
        size = size_x + pad * 2  # padding后尺寸
        if x.ndim == 4:  # 每个元素是3维的，x的0维是mini-batch
            # 初始化同维全0矩阵
            padding = np.zeros((x.shape[0], x.shape[1], size, size), dtype=Params.DTYPE_DEFAULT)
            # 中间以x填充
            padding[:, :, pad: pad + size_x, pad: pad + size_x] = x

        elif x.ndim == 3:  # 每个元素是2维的
            padding = np.zeros((x.shape[0], size, size), dtype=Params.DTYPE_DEFAULT)
            padding[:, pad: pad + size_x, pad: pad + size_x] = x

        return padding

    # 执行环境内存充裕blas方法较快
    # 否则使用jit后的np.matmul方法
    # @numba.jit
    def matmul(a, b):
        return np.matmul(a, b)

    # 输出层结果转换为标准化概率分布，
    # 入参为原始线性模型输出y ，N*K矩阵，
    # 输出矩阵规格不变
    @staticmethod
    def softmax(y):
        # 对每一行：所有元素减去该行的最大的元素,避免exp溢出,得到1*N矩阵,
        max_y = np.max(y, axis=1)
        # 极大值重构为N * 1 数组
        max_y.shape = (-1, 1)
        # 每列都减去该列最大值
        y1 = y - max_y
        # 计算exp
        exp_y = np.exp(y1)
        # 按行求和，得1*N 累加和数组
        sigma_y = np.sum(exp_y, axis=1)
        # 累加和reshape为N*1 数组
        sigma_y.shape = (-1, 1)
        # 计算softmax得到N*K矩阵
        softmax_y = exp_y / sigma_y

        return softmax_y

    # 交叉熵损失函数
    # 限制上界避免除零错
    @staticmethod
    def crossEntropy(y,y_,eps):
        return -np.log(np.clip(y[range(len(y)), y_],eps,None,None))

    # 持久化训练参数
    def traceMatrix(M, epoch, name):

        if TRACE_FLAG == False:
            return 0
        row = len(M)
        try:
            col = len(M[0])
        except TypeError:
            col = 1
        with open(trace_file, 'a') as file:
            file.write('Epoch[%s]-[%s:%d X %d ]----------------------------------------\n' % (epoch, name, row, col))
            for i in range(row):
                file.write('%s -- %s\n' % (i, M[i]))

# data loading
class MnistData(object):

    def __init__(self, absPath, is4Cnn, dataType):
        self.absPath = absPath
        self.is4Cnn = is4Cnn  # True for cnn,False for other nn structures
        self.dataType = dataType
        self.imgs, self.labels = self._load_mnist_data(kind='train')
        self.imgs_v, self.labels_v = self._load_mnist_data(kind='t10k')
        self.sample_range = [i for i in range(len(self.labels))]  # 训练样本范围
        self.sample_range_v = [i for i in range(len(self.labels_v))]  # 验证样本范围

    # 加载mnist
    def _load_mnist_data(self, kind='train'):
        labels_path = os.path.join(self.absPath, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(self.absPath, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as labelfile:
            # 读取前8个bits
            magic, n = struct.unpack('>II', labelfile.read(8))
            # 余下的数据读到标签数组中
            labels = np.fromfile(labelfile, dtype=np.uint8)

        with open(images_path, 'rb') as imagefile:
            # 读取前16个bit
            magic, num, rows, cols = struct.unpack('>IIII', imagefile.read(16))
            # 余下数据读到image二维数组中，28*28=784像素的图片共60000张（和标签项数一致）
            # reshape 从原数组创建一个改变尺寸的新数组(28*28图片拉直为784*1的数组)
            # CNN处理的输入则reshape为28*28*1
            if False == self.is4Cnn:
                images_ori = np.fromfile(imagefile, dtype=np.uint8).reshape(len(labels), 784)
            else:
                # 支持多通道，此处通道为1
                images_ori = np.fromfile(imagefile, dtype=np.uint8).reshape(len(labels), 1, 28, 28)
            # 归一化
            images = images_ori / 255
        return images, labels

    # 对训练样本序号随机分组
    def getTrainRanges(self, miniBatchSize):

        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 获取训练样本范围对应的图像和标签
    def getTrainDataByRng(self, rng):

        xs = np.array([self.imgs[sample] for sample in rng], self.dataType)
        values = np.array([self.labels[sample] for sample in rng])
        return xs, values

    # 获取随机验证样本
    def getValData(self, valCapacity):

        samples_v = random.sample(self.sample_range_v, valCapacity)
        #  验证输入 N*28*28
        images_v = np.array([self.imgs_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        #  正确类别 1*K
        labels_v = np.array([self.labels_v[sample_v] for sample_v in samples_v])

        return images_v, labels_v

# 全连接类
class FCLayer(object):
    def __init__(self, miniBatchesSize, i_size, o_size,
                 activator, optimizer,
                 dataType):
        # 初始化超参数
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.o_size = o_size
        self.activator = activator
        self.optimizer = optimizer
        self.dataType = dataType
        self.w = Params.INIT_W * np.random.randn(i_size, o_size).astype(dataType)
        self.b = np.zeros(o_size, dataType)
        self.out = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出

    # 前向传播,激活后再输出
    def inference(self, input):
        self.out = self.activator.activate(Tools.matmul(input, self.w) + self.b)
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.bpDelta()
        self.bpWeights(input, lrt)

        return self.deltaPrev

    # 输出误差反向传播至上一层
    def bpDelta(self):
        self.deltaPrev = Tools.matmul(self.deltaOri, self.w.T)
        return self.deltaPrev

    # 计算反向传播权重梯度w,b
    def bpWeights(self, input, lrt):
        dw = Tools.matmul(input.T, self.deltaOri)
        db = np.sum(self.deltaOri, axis=0, keepdims=True).reshape(self.b.shape)
        wNew, bNew = self.optimizer.getUpdWeights(self.w, dw, self.b, db, lrt)
        self.w = wNew
        self.b = bNew

# 卷积处理类
class ConvLayer(object):
    def __init__(self, LName, miniBatchesSize, i_size,
                 channel, f_size, o_depth, o_size,
                 strides, activator, optimizer,
                 dataType):

        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.channel = channel
        self.f_size = f_size
        self.o_depth = o_depth
        self.o_size = o_size
        self.strides = strides
        self.activator = activator
        self.optimizer = optimizer
        self.dataType = dataType
        self.w = Params.INIT_W * np.random.randn(o_depth, channel,
                                                 f_size, f_size).astype(self.dataType)
        self.b = np.zeros((o_depth, 1), dtype=dataType)
        self.out = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出

    # 前向传播,激活后输出
    def inference(self, input):

        self.out = self.activator.activate(self.conv_efficient(input,
                                                               self.w, self.b,
                                                               self.o_size, self.name,
                                                               self.strides))
        return self.out

    # 反向传播(误差和权参),先对输出误差求导再反向传播至上一层
    def bp(self, input, delta, lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.deltaPrev, dw, db = self.bp4conv(self.deltaOri, self.w, input,
                                              self.strides, 'd' + self.name)

        wNew, bNew = self.optimizer.getUpdWeights(self.w, dw, self.b, db, lrt)
        self.w = wNew
        self.b = bNew

        return self.deltaPrev

    # conv4dw,反向传播计算dw
    # 以卷积层输出误差为卷积核，对卷计层输入做卷积，得到卷积层w的梯度
    # 输入输出尺寸不变的过滤器 当s==1时，p=(f-1)/2
    # 入参:
    #     x规格: 根据x.ndim 判断入参的规格
    #           x.ndim=4:原始规格,未padding
    #                   batch * depth_i * row * col,  其中 depth_i为输入节点矩阵深度，
    #           x.ndim=3:x_col规格,已padding
    #                   前向:batch * (depth_i * filter_size * filter_size) * (out_size*out_size)
    #                   反向:batch * depth_i * ( filter_size * filter_size) * (out_size*out_size)
    #                   注意，反向传播时，x_col保持四个维度而不是前向传播的三个
    #     w规格: batches * depth_o  * filter_size * filter_size ， ，
    #           depth_o为过滤器个数或输出矩阵深度，
    #           w_row: batches * depth_o * ( filter_size * filter_size)
    #           此处的w是反向传播过来卷积层输出误差，没有depth_i这个维度
    #     b规格: 长度为 depth_o*1 的数组,b的长度即为过滤器个数或节点深度,和w的depth_o一致。
    #           conv4dw时,b为0
    #     output_size:conv4dw的输出矩阵尺寸,对应原始卷积层w的尺寸
    #     strides: 缺省为1
    #     x_v : False x未作矢量化,True x已作向量化(对第一层卷积适用，每个mini-batch多个Iteration时可提速)
    # 返回: 卷积层加权输出(co-relation)
    #       conv : batch * depth_o * depth_i * output_size * output_size
    def conv4dw(self, x, w, output_size, b=0, strides=1, x_v=False):
        batches = x.shape[0]
        depth_i = x.shape[1]
        filter_size = w.shape[2]  # 过滤器尺寸,对应卷积层误差矩阵尺寸
        x_per_filter = filter_size * filter_size
        depth_o = w.shape[1]

        if False == x_v:  # 原始规格:
            input_size = x.shape[2]  #
            p = int(((output_size - 1) * strides + filter_size - input_size) / 2)  # padding尺寸
            if p > 0:  # 需要padding处理
                x_pad = Tools.padding(x, p)
            else:
                x_pad = x
            logger.debug("vec4dw begin..")
            x_col = self.vectorize4convdw_batches(x_pad, filter_size, output_size, strides)
            logger.debug("vec4dw end..")
        else:  # x_col规格
            x_col = x

        w_row = w.reshape(batches, depth_o, x_per_filter)
        conv = np.zeros((batches, depth_i, depth_o, (output_size * output_size)), dtype=self.dataType)
        logger.debug("conv4dw matmul begin..")
        for batch in range(batches):
            for col in range(depth_i):
                conv[batch, col] = Tools.matmul(w_row[batch], x_col[batch, col])

        conv_sum = np.sum(conv, axis=0)
        # transpose而不是直接reshape避免错位
        conv = conv_sum.transpose(1, 0, 2).reshape(depth_o, depth_i, output_size, output_size)

        logger.debug("conv4dw matmul end..")
        return conv, x_col

    # conv_efficient,使用向量化和BLAS优化的卷积计算版本
    # 入参:
    #     x规格: 根据x.ndim 判断入参的规格
    #           x.ndim=4:原始规格,未padding
    #                   batch * depth_i * row * col,  其中 depth_i为输入节点矩阵深度，
    #           x.ndim=3:x_col规格,已padding
    #                   batch * (depth_i * filter_size * filter_size) * (out_size*out_size)
    #     w规格: depth_o * depth_i * filter_size * filter_size ， ，
    #           depth_o为过滤器个数或输出矩阵深度，depth_i和 x的 depth一致
    #           w_row: depth_o * (depth_i * filter_size * filter_size)
    #     b规格: 长度为 depth_o*1 的数组,b的长度即为过滤器个数或节点深度,和w的depth_o一致，可以增加校验。
    #     output_size:卷积输出尺寸
    #     strides: 缺省为1
    #     vec_idx_key: vec_idx键
    # 返回: 卷积层加权输出(co-relation)
    #       conv : batch * depth_o * output_size * output_size
    def conv_efficient(self, x, w, b, output_size, vec_idx_key, strides=1):
        batches = x.shape[0]
        depth_i = x.shape[1]
        filter_size = w.shape[2]
        depth_o = w.shape[0]

        if 4 == x.ndim:  # 原始规格:
            input_size = x.shape[2]  #
            p = int(((output_size - 1) * strides + filter_size - input_size) / 2)  # padding尺寸
            # logger.debug("padding begin..")
            if p > 0:  # 需要padding处理
                x_pad = Tools.padding(x, p)
            else:
                x_pad = x
            st = time.time()
            logger.debug("vecting begin..")
            # 可以根据自己的硬件环境，在三种优化方式中选择较快的一种
            x_col = self.vectorize4conv_batches(x_pad, filter_size, output_size, strides)
            #x_col = spd.vectorize4conv_batches(x_pad, filter_size, output_size, strides)
            #x_col = vec_by_idx(x_pad, filter_size, filter_size,vec_idx_key,0, strides)

            logger.debug("vecting end.. %f s" % (time.time() - st))
        else:  # x_col规格
            x_col = x

        w_row = w.reshape(depth_o, x_col.shape[1])
        conv = np.zeros((batches, depth_o, (output_size * output_size)), dtype=self.dataType)
        st1 = time.time()
        logger.debug("matmul begin..")
        #不广播，提高处理效率
        for batch in range(batches):
            conv[batch] = Tools.matmul(w_row, x_col[batch]) + b

        logger.debug("matmul end.. %f s" % (time.time() - st1))
        conv_return = conv.reshape(batches, depth_o, output_size, output_size)

        return conv_return

    # vectorize4convdw_batches:用于反向传播计算dw的向量化
    # ------------------------------------
    # 入参
    #    x : padding后的实例 batches * channel * conv_i_size * conv_i_size
    #    fileter_size :
    #    conv_o_size:
    #    strides:
    # 返回
    #    x_col: batches *channel* (filter_size * filter_size) * ( conv_o_size * conv_o_size)
    # @numba.jit
    def vectorize4convdw_batches(self, x, filter_size, conv_o_size, strides):
        batches = x.shape[0]
        channels = x.shape[1]
        x_per_filter = filter_size * filter_size
        x_col = np.zeros((batches, channels, x_per_filter, conv_o_size * conv_o_size), dtype=self.dataType)
        for j in range(x_col.shape[3]):
            b = int(j / conv_o_size) * strides
            c = (j % conv_o_size) * strides
            x_col[:, :, :, j] = x[:, :, b:b + filter_size, c:c + filter_size].reshape(batches, channels, x_per_filter)

        return x_col

    # cross-correlation向量化优化
    # x_col = (depth_i * filter_size * filter_size) * (conv_o_size * conv_o_size)
    # w: depth_o * （ depth_i/channel * conv_i_size * conv_o_size） =  2*3*3*3
    # reshape 为 w_row =  depth_o * (depth_i/channel * (conv_i_size * conv_o_size)) = 2 * 27
    # conv_t= matmul(w_row,x_col)
    # 得到 conv_t = depth_o * (conv_o_size * conv_size) = 2 * (3*3) =2*9
    #  再 conv = conv_t.reshape ( depth_o * conv_o_size * conv_size) = (2*3*3)
    # （只看一个filter则有九个点乘和，每个点乘和来自深度为3的x的三个九宫格，把27格展开reshape成一列
    # ------------------------------------
    # 入参
    #    x : padding后的实例 batches * channel * conv_i_size * conv_i_size
    #    fileter_size :
    #    conv_o_size:
    #    strides:
    # 返回
    #    x_col: batches *(channel* filter_size * filter_size) * ( conv_o_size * conv_o_size)
    # @numba.jit
    def vectorize4conv_batches(self, x, filter_size, conv_o_size, strides):
        batches = x.shape[0]
        channels = x.shape[1]
        x_per_filter = filter_size * filter_size
        shape_t = channels * x_per_filter
        x_col = np.zeros((batches, channels * x_per_filter, conv_o_size * conv_o_size), dtype=self.dataType)
        for j in range(x_col.shape[2]):
            b = int(j / conv_o_size) * strides
            c = (j % conv_o_size) * strides
            x_col[:, :, j] = x[:, :, b:b + filter_size, c:c + filter_size].reshape(batches, shape_t)

        return x_col

    # bp4conv: conv反向传播梯度计算
    # 入参:
    #    d_o :卷积输出误差 batches * depth_o * output_size * output_size   ，规格同 conv的输出
    #    w: depth_o * depth_i * filter_size * filter_size
    #    a: 原卷积层输入 batch * depth_i * input_size * input_size
    #    strides:
    # 返参:
    #    d_i :卷积输入误差 batch * depth_i * input_size * input_size,  其中 depth_i为输入节点矩阵深度
    #    dw : w梯度，规格同w
    #    db : b 梯度 规格同b, depth_O * 1 数组
    #    vec_idx_key:
    # 说明: 1.误差反向传递和db
    #      将w翻转180度作为卷积核，
    #      在depth_o上，对每一层误差矩阵14*14，以该层depth_i个翻转后的w 5*5,做cross-re得到 depth_i个误差矩阵14*14
    #      所有depth_o做完，得到depth_o组，每组depth_i个误差矩阵
    #           batch * depth_o * depth_i * input_size * input_size
    #      d_i:每组同样位置的depth_o个误差矩阵相加，得到depth_i个误差矩阵d_i ,规格同a
    #          优化, 多维数组w_rtLR， 在dept_o和dept_i上做转置，作为卷积和与d_o组协相关
    #      db: 每个d_o上的误差矩阵相加
    #     2. dw
    #       以d_o作为卷积核，对原卷积层输入a做cross-correlation得到 dw
    #       do的每一层depth_o，作为卷积核 14*14，
    #                       与原卷积的输入a的每一个depth_i输入层14*14和做cross-re 得到,depth_i个结果矩阵5*5
    #               合计depth_o * depth_i * f_size * f_size
    #                       只要p/s =2 即可使结果矩阵和w同样规格，如 p=2,s=1
    #               每个结果矩阵作为该depth_o上，该输入层w对应的dw。
    def bp4conv(self, d_o, w, a, strides, vec_idx_key):
        st = time.time()
        logger.debug("bp4conv begin..")
        input_size = a.shape[2]
        f_size = w.shape[2]

        # w翻转180度,先上下翻转，再左右翻转，然后前两维互换实现高维转置
        w_rtUD = w[:, :, ::-1]  # 上下翻转
        w_rtLR = w_rtUD[:, :, :, ::-1]  # 左右翻转
        w_rt = w_rtLR.transpose(1, 0, 2, 3)  # 0维和1维互换实现高维转置

        # 误差项传递
        d_i = self.conv_efficient(d_o, w_rt, 0, input_size, vec_idx_key, 1)
        logger.debug("d_i ready..")

        # 每个d_o上的误差矩阵相加
        db = np.sum(np.sum(np.sum(d_o, axis=-1), axis=-1), axis=0).reshape(-1, 1)
        logger.debug("db ready.. %f s" % (time.time() - st))

        dw, x_col = self.conv4dw(a, d_o, f_size, 0, 1, False)
        logger.debug("bp4conv end.. %f s" % (time.time() - st))

        return d_i, dw, db

    # 梯度检查todo
    def init_test():
        pass

# 池化处理类
class MaxPoolLayer(object):
    def __init__(self, LName, miniBatchesSize, f_size,
                 strides, needReshape, dataType):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.f_size = f_size
        self.strides = strides
        self.needReshape = needReshape  # 输出到全连接层是否需要reshape
        self.dataType = dataType
        self.out = []
        self.shapeOfOriOut = ()  # 保留原始输出的shape，用于反向传播
        self.poolIdx = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出

    # 前向传播
    def inference(self, input):

        pooling, self.poolIdx = self.pool(input, self.f_size, self.strides, 'MAX')
        self.shapeOfOriOut = pooling.shape
        if True == self.needReshape:
            self.out = pooling.reshape(pooling.shape[0], -1)
        else:
            self.out = pooling
        return self.out

    # 反向传播,有误差无权参，先对输出误差求导再反向传播至上一层
    def bp(self, input, delta, lrt):

        if True == self.needReshape:
            self.deltaOri = delta.reshape(self.shapeOfOriOut)
        else:
            self.deltaOri = delta

        self.deltaPrev = self.bp4pool(self.deltaOri, self.poolIdx,
                                      self.f_size, self.strides, 'MAX')

        return self.deltaPrev

    # pooling, 优化后的的降采样计算
    # 入参:
    #     x规格: batch * depth_i * row * col,  其中 depth_i为输入节点矩阵深度，
    #     fileter_size: 过滤器尺寸
    #     strides: 缺省为1
    #     type: 降采样类型,MAX/MEAN  ,缺省为MAX
    # 返回: 卷积层加权输出(co-relation)
    #       pooling : batch * depth_i * output_size * output_size
    #       pooling_idx : batch * depth_i * y_per_o_layer * x_per_filter
    #                其中 y_per_o_layer =  output_size * output_size
    #                    x_per_filter = pool_f_size * pool_f_size
    #                MAX value在当前input_block 对应位置为1,其它为0
    # 优化：先把x 组织成 batch * depth_i * (output_size * output_size) * (filter_size * filter_size)
    #      然后利用矩阵运算，对最后一维做max得到batch * depth_i * (output_size * output_size)
    #      再reshape 为 batch * depth_i * output_size * output_size
    def pool(self, x, filter_size, strides=2, type='MAX'):
        logger.debug("pooling begin..")
        batches = x.shape[0]
        depth_i = x.shape[1]
        input_size = x.shape[2]  #
        x_per_filter = filter_size * filter_size
        output_size = int((input_size - filter_size) / strides) + 1
        y_per_o_layer = output_size * output_size  # 输出矩阵,每一层元素个数
        x_vec = np.zeros((batches, depth_i, y_per_o_layer, x_per_filter), dtype=self.dataType)

        # pooling处理
        for j in range(y_per_o_layer):
            b = int(j / output_size) * strides
            c = (j % output_size) * strides
            x_vec[:, :, j, 0:x_per_filter] = x[:, :, b:b + strides, c:c + strides].reshape(batches, depth_i,
                                                                                           x_per_filter)

        pooling = np.max(x_vec, axis=3).reshape(batches, depth_i, output_size, output_size)
        pooling_idx = np.eye(x_vec.shape[3], dtype=int)[x_vec.argmax(3)]
        logger.debug("pooling end..")

        return pooling, pooling_idx

    # bp4pool: 反向传播上采样梯度
    # 入参：
    #      dpool: 池化层输出的误差项, N * 3136 =N*(64*7*7)=  batches * (depth_i * pool_o_size * pool_o_size)
    #                  reshape为batches * depth_i * pool_o_size * pool_o_size
    #      pool_idx : MAX pool时保留的max value index , batches * depth_i * y_o * x_per_filter
    #      pool_f_size: pool  filter尺寸
    #      pool_stides:
    #      type : MAX ,MEAN, 缺省为MAX
    # 返参:
    #      dpool_i: 传递到上一层的误差项  , batches * depth_i * pool_i_size * pool_i_size
    #             当 strides =2 ,filter = 2 时， pool的pool_i_size 是pool_o_size 的2倍
    def bp4pool(self, dpool, pool_idx, pool_f_size, pool_strides, type='MAX'):
        logger.debug("bp4pool begin..")
        batches = dpool.shape[0]
        depth_i = pool_idx.shape[1]
        y_per_o = pool_idx.shape[2]

        x_per_filter = pool_f_size * pool_f_size
        pool_o_size = int(np.sqrt(y_per_o))

        input_size = (pool_o_size - 1) * pool_strides + pool_f_size
        dpool_reshape = dpool.reshape(batches, depth_i, y_per_o)

        dpool_i_tmp = np.zeros((batches, depth_i, input_size, input_size), dtype=self.dataType)
        pool_idx_reshape = np.zeros(dpool_i_tmp.shape, dtype=self.dataType)
        for j in range(y_per_o):
            b = int(j / pool_o_size) * pool_strides
            c = (j % pool_o_size) * pool_strides
            # pool_idx_reshape规格同池化层输入，每个block的max value位置值为1，其余位置值为0
            pool_idx_reshape[:, :, b:b + pool_f_size, c:c + pool_f_size] = pool_idx[:, :, j, 0:x_per_filter].reshape(
                batches,
                depth_i,
                pool_f_size,
                pool_f_size)
            # dpool_i_tmp规格规格同池化层输入，每个block的值均以对应dpool元素填充
            for row in range(pool_f_size):  # 只需要循环 x_per-filter 次得到 填充扩展后的delta
                for col in range(pool_f_size):
                    dpool_i_tmp[:, :, b + row, c + col] = dpool_reshape[:, :, j]
        # 相乘后，max value位置delta向上传播，其余位置为delta为0
        dpool_i = dpool_i_tmp * pool_idx_reshape
        logger.debug("bp4pool end..")
        return dpool_i


# 自适应矩估计优化类
class AdmOptimizer(object):
    def __init__(self, beta1, beta2, eps, dataType):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dataType = dataType
        self.isInited = False
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.Iter = 0

    # lazy init
    def initMV(self, shapeW, shapeB):
        if (False == self.isInited):
            self.m_w = np.zeros(shapeW, dtype=self.dataType)
            self.v_w = np.zeros(shapeW, dtype=self.dataType)
            self.m_b = np.zeros(shapeB, dtype=self.dataType)
            self.v_b = np.zeros(shapeB, dtype=self.dataType)
            self.isInited = True

    def getUpdWeights(self, w, dw, b, db, lr):
        self.initMV(w.shape, b.shape)

        t = self.Iter + 1
        wNew, self.m_w, self.v_w = self.OptimzAdam(w, dw, self.m_w, self.v_w, lr, t)
        bNew, self.m_b, self.v_b = self.OptimzAdam(b, db, self.m_b, self.v_b, lr, t)
        self.Iter += 1
        return wNew, bNew

    def OptimzAdam(self, x, dx, m, v, lr, t):
        beta1 = self.beta1
        beta2 = self.beta2
        m = self.beta1 * m + (1 - self.beta1) * dx
        mt = m / (1 - self.beta1 ** t)
        v = self.beta2 * v + (1 - self.beta2) * (dx ** 2)
        vt = v / (1 - self.beta2 ** t)
        x += - lr * mt / (np.sqrt(vt) + self.eps)

        return x, m, v

# ReLU  Activator
class ReLU(object):
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def bp(delta, x):
        delta[x <= 0] = 0
        return delta

# Pass Activator
class NoAct(object):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def bp(delta, x):
        return delta

# 训练完成后，生成训练过程图像
class ResultView(object):
    def __init__(self, epoch, line_labels, colors, ax_labels, dataType):
        self.cur_p_idx = 0
        self.curv_x = np.zeros(epoch * 100, dtype=int)
        self.curv_ys = np.zeros((4, epoch * 100), dtype=dataType)
        self.line_labels = line_labels
        self.colors = colors
        self.ax_labels = ax_labels

    def addData(self, curv_x, loss, loss_v, acc, acc_v):

        self.curv_x[self.cur_p_idx] = curv_x
        self.curv_ys[0][self.cur_p_idx] = loss
        self.curv_ys[1][self.cur_p_idx] = loss_v
        self.curv_ys[2][self.cur_p_idx] = acc
        self.curv_ys[3][self.cur_p_idx] = acc_v
        self.cur_p_idx += 1

    # 显示曲线
    def show(self):
        self.showCurves(self.cur_p_idx, self.curv_x, self.curv_ys, self.line_labels, self.colors, self.ax_labels)

    def showCurves(self, idx, x, ys, line_labels, colors, ax_labels):
        LINEWIDTH = 2.0
        plt.figure(figsize=(8, 4))
        # loss
        ax1 = plt.subplot(211)
        for i in range(2):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax1.xaxis.set_major_locator(MultipleLocator(4000))
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.set_xlabel(ax_labels[0])
        ax1.set_ylabel(ax_labels[1])
        plt.grid()
        plt.legend()

        # Acc
        ax2 = plt.subplot(212)
        for i in range(2, 4):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax2.xaxis.set_major_locator(MultipleLocator(4000))
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.set_xlabel(ax_labels[0])
        ax2.set_ylabel(ax_labels[2])

        plt.grid()
        plt.legend()
        plt.show()

# NN会话类
class Session(object):
    # 构造各层
    def __init__(self, layers):
        self.layers = layers
        self.input = []

    # 前向传播和验证
    def inference(self, train_data, y_):
        curr_batch_size = len(y_)
        self.input = train_data
        dataLayer = train_data
        for layer in self.layers:
            dataLayer = layer.inference(dataLayer)

        ## acc
        y = np.argmax(dataLayer, axis=1)
        acc_t = np.mean(y == y_)
        # 最后做softmax
        softmax_y = Tools.softmax(dataLayer)

        # loss
        corect_logprobs = Tools.crossEntropy(softmax_y, y_, Params.EPS2)
        data_loss = np.sum(corect_logprobs) / curr_batch_size

        # delta
        softmax_y[range(curr_batch_size), y_] -= 1
        delta = softmax_y / curr_batch_size
        return acc_t, data_loss, delta

    # 逐层前向传播

    def bp(self, delta, lrt):
        deltaLayer = delta
        for i in reversed(range(1, len(self.layers))):
            deltaLayer = self.layers[i].bp(self.layers[i - 1].out, deltaLayer, lrt)

        # 第一层，以输入作为输出,误差不再反向传播
        self.layers[0].bp(self.input, deltaLayer, lrt)

    # 实现训练步骤
    def train_steps(self, train_data, y_, lrt):
        acc, loss, delta = self.inference(train_data, y_)
        self.bp(delta, lrt)
        return acc, loss

    # 独立数据集验证训练结果
    def validation(self, data_v, y_v):
        acc, loss, _ = self.inference(data_v, y_v)
        return acc, loss

def main():
    logger.info('start..')
    # 初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    if (True == Params.SHOW_LOSS_CURVE):
        view = ResultView(Params.EPOCH_NUM,
                          ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                          ['y', 'r', 'g', 'b'],
                          ['Iteration', 'Loss', 'Accuracy'],
                          Params.DTYPE_DEFAULT)
    # time stamp
    s_t = 0

    # 数据对象初始化
    mnist = MnistData(path_minst_unpack, True, Params.DTYPE_DEFAULT)

    # 定义网络结构，支持各层使用不同的优化方法。
    # 输入层->卷积层1->池化层1->卷积层2->池化层2->FC1->FC2->softmax->输出结果
    conv1Optimizer = AdmOptimizer(Params.BETA1, Params.BETA2, Params.EPS, Params.DTYPE_DEFAULT)
    conv1 = ConvLayer('conv1', Params.MINI_BATCH_SIZE, Params.IMAGE_SIZE, Params.IMAGE_CHANNEL,
                      Params.CONV1_F_SIZE, Params.CONV1_O_DEPTH,
                      Params.CONV1_O_SIZE, Params.CONV1_STRIDES,
                      ReLU, conv1Optimizer, Params.DTYPE_DEFAULT)
    pool1 = MaxPoolLayer('pool1', Params.MINI_BATCH_SIZE, Params.POOL1_F_SIZE,
                         Params.POOL1_STRIDES, False, Params.DTYPE_DEFAULT)

    conv2Optimizer = AdmOptimizer(Params.BETA1, Params.BETA2, Params.EPS, Params.DTYPE_DEFAULT)
    # conv2 = ConvLayer('conv2', Params.MINI_BATCH_SIZE, Params.IMAGE_SIZE, Params.CONV1_O_DEPTH,
    conv2 = ConvLayer('conv2', Params.MINI_BATCH_SIZE, Params.CONV2_O_SIZE, Params.CONV1_O_DEPTH,
                      Params.CONV2_F_SIZE, Params.CONV2_O_DEPTH,
                      Params.CONV2_O_SIZE, Params.CONV2_STRIDES,
                      ReLU, conv2Optimizer, Params.DTYPE_DEFAULT)
    pool2 = MaxPoolLayer('pool2', Params.MINI_BATCH_SIZE, Params.POOL2_F_SIZE,
                         Params.POOL2_STRIDES, True, Params.DTYPE_DEFAULT)

    fc1Optimizer = AdmOptimizer(Params.BETA1, Params.BETA2, Params.EPS, Params.DTYPE_DEFAULT)
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.FC1_SIZE_INPUT, Params.FC1_SIZE_OUTPUT, ReLU,
                  fc1Optimizer, Params.DTYPE_DEFAULT)

    fc2Optimizer = AdmOptimizer(Params.BETA1, Params.BETA2, Params.EPS, Params.DTYPE_DEFAULT)
    fc2 = FCLayer(Params.MINI_BATCH_SIZE, Params.FC1_SIZE_OUTPUT, Params.TYPE_K, NoAct,
                  fc2Optimizer, Params.DTYPE_DEFAULT)

    cnnLayers = [conv1, pool1, conv2, pool2, fc1, fc2]

    # 生成训练会话实例
    sess = Session(cnnLayers)

    # 开始训练过程
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        for key in Params.DIC_L_RATE.keys():
            if (epoch + 1) < key:
                break
            lrt = Params.DIC_L_RATE[key]

        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = mnist.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y_ = mnist.getTrainDataByRng(dataRngs[batch])
            acc_t, loss_t = sess.train_steps(x, y_, lrt)

            if (batch % 10 == 0):  # 10个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f,acc_t= %.3f st[%.1f]" % (
                    epoch, batch, loss_t, acc_t, s_t))

            # 使用随机验证样本验证结果
            if (batch % 30 == 0 and (batch+epoch) >0):
                x_v, y_v = mnist.getValData(Params.VALIDATION_CAPACITY)
                acc_v, loss_v = sess.validation(x_v, y_v)

                logger.info('epoch %2d-%3d, loss=%f, loss_v=%f, acc=%f, acc_v=%f' % (
                    epoch, batch, loss_t, loss_v, acc_t, acc_v))
                # 可视化记录
                if (True == Params.SHOW_LOSS_CURVE):
                    view.addData(fc1Optimizer.Iter,
                                 loss_t, loss_v, acc_t, acc_v)

            s_t = time.time() - start
            s_t = time.time() - start


    logger.info('session end')
    view.show()

if __name__ == '__main__':
    main()
