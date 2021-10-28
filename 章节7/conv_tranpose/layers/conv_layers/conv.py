import tensorflow.compat.v1 as tf
import math


def conv2d(inp, 
           out_channel, 
           kernel, 
           stride, 
           padding='SAME', 
           use_bias=False, 
           use_fan_in=True, 
           name='conv'):
    with tf.variable_scope(name):
        # 默认卷积的输入都是4维的张量，格式为NHWC
        # 从最后一个维度拿出C_{in}
        _, _, _, C = inp.get_shape().as_list()
        
        # 根据use_fan_in的值选择使用输入或输出的数量作为参数初始化方差
        fan_in = kernel * kernel * C
        fan_out = kernel * kernel * out_channel
        fan_num = fan_in if use_fan_in else fan_out
        
        w = tf.Variable(
                tf.truncated_normal(
                    [kernel, kernel, C, out_channel], 
                    mean=0.0, stddev=math.sqrt(2 / fan_num)
                ), name='w')

        output = tf.nn.conv2d(inp, 
                              filter=w, 
                              strides=[1, stride, stride, 1], 
                              padding=padding, 
                              name='conv')

        # 是否使用偏置
        if use_bias:
            b = tf.Variable(tf.zeros([out_channel]), name='b')
            output = tf.add(output, b)

        return output

def conv2d_(inp, 
            out_channel, 
            kernel, 
            stride, 
            padding='SAME', 
            use_bias=True, 
            use_fan_in=True, 
            name='conv'):
    with tf.variable_scope(name):
        return tf.layers.conv2d(
                inp,
                out_channel,
                kernel,
                stride,
                padding=str.lower(padding),
                use_bias=use_bias,
                name='conv'
        )

def depthwise_conv2d(inp, 
                     kernel, 
                     channel_multiplier, 
                     stride, 
                     padding='SAME', 
                     dilation=1, 
                     use_bias=False, 
                     use_fan_in=True, 
                     name='depthwise_conv'):
    with tf.variable_scope(name):
        _, _, _, C = inp.get_shape().as_list()
        
        fan_in = kernel * kernel * C
        fan_out = kernel * kernel * C * channel_multiplier
        fan_num = fan_in if use_fan_in else fan_out

        # 使用Kaiming初始化卷积核参数
        w = tf.Variable(
                tf.truncated_normal(
                    [kernel, kernel, C, channel_multiplier],
                     mean=0.0, stddev=math.sqrt(2 / fan_num)), 
                name='w')

        # 为tf.nn.depthwise_conv2d传入参数执行深度卷积
        output = tf.nn.depthwise_conv2d(inp, 
                                        filter=w, 
                                        strides=[1, stride, stride, 1], 
                                        padding=padding, 
                                        rate=[dilation, dilation], 
                                        name='depthwise_conv')

        # 是否使用Bias，一般不使用
        if use_bias:
            b = tf.Variable(tf.zeros([C * channel_multiplier]), name='b')
            output = tf.add(output, b)

        return output

def depth_separable_conv2d(inp, 
                           depth_kernel, 
                           channel_multiplier, 
                           out_channel, 
                           stride, 
                           padding='SAME', 
                           use_bias=False, 
                           use_fan_in=True, 
                           name='sep_conv'):
    with tf.variable_scope(name):
        _, _, _, C = inp.get_shape().as_list()
        
        # 深度卷积后的输出特征的通道数
        depth_out_channel = int(C * channel_multiplier)

        # 对深度卷积的卷积核使用Kaiming初始化
        depth_fan_in = depth_kernel * depth_kernel * C
        depth_fan_out = depth_kernel * depth_kernel * depth_out_channel
        depth_fan_num = depth_fan_in if use_fan_in else depth_fan_out

        # 对逐点卷积的卷积核使用Kaiming初始化
        point_fan_in = depth_out_channel
        point_fan_out = out_channel
        point_fan_num = point_fan_in if use_fan_in else point_fan_out

        # 深度卷积的卷积核
        depth_filter = tf.Variable(
                        tf.truncated_normal(
                            [depth_kernel, depth_kernel, C, channel_multiplier], 
                            mean=0.0, stddev=math.sqrt(2 / depth_fan_num)),
                        name='depth_w')
        
        # 逐点卷积的卷积核（1*1卷积）
        point_filter = tf.Variable(
                            tf.truncated_normal(
                                [1, 1, channel_multiplier, out_channel], 
                                mean=0.0, stddev=math.sqrt(2 / point_fan_num)), 
                            name='point_w')

        # 执行深度可分离卷积
        output = tf.nn.separable_conv2d(inp, 
                                        depth_filter, 
                                        point_filter, 
                                        strides=[1, stride, stride, 1], 
                                        rate=[1, 1], 
                                        padding=padding, 
                                        name='depth_point_conv')

        if use_bias:
            b = tf.Varable(tf.zeros([out_channel]), name='b')
            output = tf.add(output, b)

        return output


def conv2d_transpose(inp, 
                     out_channel,
                     out_size, 
                     kernel, 
                     stride, 
                     padding='SAME', 
                     use_bias=False, 
                     use_fan_in=True, 
                     name='conv_transpose'):
    with tf.variable_scope(name):
        # 默认卷积的输入都是4维的张量，格式为NHWC
        # 从最后一个维度拿出C_{in}
        N, _, _, C = inp.get_shape().as_list()
        
        # 根据use_fan_in的值选择使用输入或输出的数量作为参数初始化方差
        fan_in = kernel * kernel * C
        fan_out = kernel * kernel * out_channel
        fan_num = fan_in if use_fan_in else fan_out
        
        w = tf.Variable(
                tf.truncated_normal(
                    [kernel, kernel, out_channel, C],
                    mean=0.0, stddev=math.sqrt(2 / fan_num)
                ), name='w')

        output = tf.nn.conv2d_transpose(inp, 
                                        filter=w,
                                        output_shape=[N, out_size, out_size, out_channel],
                                        strides=[1, stride, stride, 1], 
                                        padding=padding, 
                                        name='conv_transpose')

        # 是否使用偏置
        if use_bias:
            b = tf.Variable(tf.zeros([out_channel]), name='b')
            output = tf.add(output, b)

        return output

def conv2d_transpose_(inp, 
                      out_channel, 
                      out_size,
                      kernel, 
                      stride, 
                      padding='SAME', 
                      use_bias=True, 
                      use_fan_in=True, 
                      name='conv_transpose'):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(
                inp,
                out_channel,
                kernel,
                stride,
                padding=str.lower(padding),
                use_bias=use_bias,
                name='conv_transpose'
        )

if __name__ == "__main__":
    x = tf.ones([1, 3, 3, 1])
    kernel = 3
    stride = 2

    out1 = conv2d_transpose(x, 3, 5, kernel, stride)
    out2 = conv2d_transpose(x, 3, 6, kernel, stride)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(out1).shape)
        print(sess.run(out2).shape)