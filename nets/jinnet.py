
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, DepthwiseConv2D, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# 普通卷积块: 标准的卷积+标准化+激活函数
def Conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


# 可分离卷积块: 利用更少的参数代替普通3x3卷积
def Depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    # 3x3可分离卷积
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 1x1普通卷积
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def JinNet(inputs, embedding_size=128, dropout_keep_prob=0.4, alpha=1.0, depth_multiplier=1):
    # 160,160,3 -> 80,80,32， 压缩像素， 拉长通道
    x = Conv_block(inputs, 32, strides=(2, 2))
    
    # 80,80,32 -> 80,80,64， 默认步长(1,1)
    x = Depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 80,80,64 -> 40,40,128
    x = Depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    x = Depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 40,40,128 -> 20,20,256
    x = Depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = Depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 20,20,256 -> 10,10,512
    x = Depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 10,10,512 -> 5,5,1024
    x = Depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = Depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    
    # 5,5,1024 -> 128特征向量
    # 1024
    x = GlobalAveragePooling2D()(x)
    # Dropout层
    # 防止网络过拟合，训练的时候起作用
    # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
    # 注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
    # 全连接层到128
    x = Dense(embedding_size, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='BatchNorm_Bottleneck')(x)
 
    # 建模型
    model = Model(inputs, x, name='mobilenet')

    return model
