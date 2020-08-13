import tensorflow as tf 

class Layers(object):
    def __init__(self):
        self.params ,self.trainable_params = [], []
        self.num_params = 0
        self.xavier_initializer = tf.initializers.glorot_normal()
    
    def elu(self,inputs): return tf.nn.elu(inputs)
    def relu(self,inputs): return tf.nn.relu(inputs)
    def sigmoid(self,inputs): return tf.nn.sigmoid(inputs)
    def softmax(self, inputs): return tf.nn.softmax(inputs)
    def dropout(self,inputs,rate): return tf.nn.dropout(inputs,rate=rate)
    
    def maxpool(self,inputs, pool_size, stride_size):
        return tf.nn.max_pool2d(inputs,ksize = [1,pool_size,pool_size,1], 
            strides = [1,stride_size,stride_size,1], padding = 'VALID')
    
    def avgpool(self,inputs,pool_size,stride_size):
        return tf.nn.avg_pool2d(inputs,ksize = [1,pool_size,pool_size,1], 
            strides = [1,stride_size,stride_size,1], padding = 'VALID')
    
    def get_weight(self,vshape,transpose = False, bias = True, name = ""):
        
        try: 
            weight_index = self.params.index("{}_w".format(name))
            if(bias): bias_index = self.params.index("{}_b".format(name))
        except:
            weight = tf.Variable(self.xavier_initializer(vshape), name = "%s_b"%(name), trainable=True, dtype=tf.float32)
            self.params.append("%s_w"%(name))
            self.trainable_params.append(weight)
            
            tmp = 1
            for i in vshape: tmp *= i
            
            self.num_params += tmp
            
            if(bias):
                if(transpose): b = tf.Variable(self.xavier_initializer([vshape[-2]]), name="%s_b"%(name), trainable=True, dtype=tf.float32)
                else: b = tf.Variable(self.xavier_initializer([vshape[-1]]),name="%s_b"%(name), trainable=True, dtype=tf.float32)
                self.params.append("%s_b"%(name))
                self.trainable_params.append(b)
                
                self.num_params += vshape[-2]
        else:
            weight = self.trainable_params[weight_index]
            if(bias): b = self.trainable_params[bias_index]
        
        if(bias): return weight,b
        else: return weight
        
        
    def fullcon(self,inputs,variables):
        [weights,biasis] = variables
        out = tf.matmul(inputs,weights) + biasis
        return out
    
    def conv2d(self,inputs,variables,stride_size, padding):
        [weights,biasis] = variables
        out = tf.nn.conv2d(inputs,weights,strides = [1,stride_size,stride_size,1],padding=padding) + biasis
        return out
    
    def batch_norm(self,inputs,name=""):
        
        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        var = std**2
        
        try:
            offset_index = self.params.index("%s_offset"%(name))
            scale_index = self.params.index("%s_scale"%(name))
        except:
            offset = tf.Variable(0, name="%s_offset"%(name), trainable = True, dtype=tf.float32)
            self.params.append("%s_offset"%(name))
            self.trainable_params.append(offset)
            self.num_params += 1
            
            scale = tf.Variable(1, name="%s_scale"%(name), trainable = True, dtype=tf.float32)
            self.params.append("%s_scale"%(name))
            self.trainable_params.append(scale)
            self.num_params += 1
        else:
            offset = self.trainable_params[offset_index]
            scale = self.trainable_params[scale_index]
        
        out = tf.nn.batch_normalization(
            x = inputs,
            mean = mean,
            variance = var,
            offset = offset,
            scale = scale,
            variance_epsilon = 1e-13,
            name = name
        )
        return out
            