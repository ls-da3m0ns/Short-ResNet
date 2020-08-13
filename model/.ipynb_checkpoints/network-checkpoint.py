import os
import tensorflow as tf 
import model.layers as lay 

class CNN(object):
    def __init__(self,height,width,channel,num_classes,ksize,learning_rate = 1e-3,chkpt_dir = './Checkpoint'):
        print("Initializing Short-ResNet .. ")
        self.height,self.width,self.channels,self.num_classes = height,width,channel,num_classes
        self.learning_rate, self.ksize = learning_rate,ksize
        self.chkpt_dir = chkpt_dir
        self.customLayers = lay.Layers()
        
        self.model(tf.zeros([1,self.height,self.width,self.channels]),verbose = True)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        
        self.summary_writer = tf.summary.create_file_writer(self.chkpt_dir)
        
    def step(self,x,y,iteration = 0,train = False):
        
        with tf.GradientTape() as tape:
            logits = self.model(x,verbose = False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits)
            loss = tf.math.reduce_mean(smce)
        
        score = self.customLayers.softmax(logits)
        pred = tf.argmax(score,1)
        correct_pred = tf.equal(pred , tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        
        if(train):
            gradients = tape.gradient(loss,self.customLayers.trainable_params)
            self.optimizer.apply_gradients(zip(gradients, self.customLayers.trainable_params))
            
            with self.summary_writer.as_default():
                tf.summary.scalar('ResNet/loss',loss,step=iteration)
                tf.summary.scalar('ResNet/accuracy',accuracy,step=iteration)
        
        return loss,accuracy,score
    
    def save_params(self):
        vars_to_save = {}
        for index,name in enumerate(self.customLayers.params):
            vars_to_save[self.customLayers.params[index]] = self.customLayers.trainable_params[index]
        vars_to_save["optimizer"] = self.optimizer
        
        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.chkpt_dir, max_to_keep = 3)
        ckptman.save()
    
    def load_params(self):
        vars_to_load = {}
        
        for index,name in enumerate(self.customLayers.params):
            vars_to_load[self.customLayers.params[index]] = self.customLayers.trainable_params[index]
        vars_to_load["optimizer"] = self.optimizer
        
        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()
        
    def model(self,x,verbose=False):
        if(verbose): print("input ",x.shape)
        
        conv1 = self.customLayers.conv2d(x,self.customLayers.get_weight(vshape = [3,3, self.channels, 16],
                                                                    name ="%s"%("conv1")),
                                        stride_size =1, padding='SAME')
        conv1_bn = self.customLayers.batch_norm(conv1,name="%s_bn"%("conv1"))
        conv1_act = self.customLayers.elu(conv1_bn)
        conv1_pool = self.customLayers.maxpool(conv1_act, pool_size=2, stride_size=2)
        
        conv2_1 = self.residual(conv1_pool,ksize=self.ksize,inchannel=16,outchannel=32,name="conv2_1",verbose=verbose)
        conv2_2 = self.residual(conv2_1,ksize=self.ksize,inchannel=32,outchannel=32,name="conv2_2",verbose=verbose)
        conv2_pool = self.customLayers.maxpool(conv2_2, pool_size=2, stride_size=2)
        
        conv3_1 = self.residual(conv2_pool,ksize=self.ksize,inchannel=32,outchannel=64,name="conv3_1",verbose=verbose)
        conv3_2 = self.residual(conv3_1,ksize=self.ksize,inchannel=64,outchannel=64,name="conv3_2",verbose=verbose)
        
        [n,h,w,c] = conv3_2.shape
        flat = tf.compat.v1.reshape(conv3_2, shape=[-1,h*w*c], name="flat")
        
        d1 = self.customLayers.fullcon(flat,self.customLayers.get_weight(vshape=[h*w*c,1024],
                                                                         name="d1"))
        d2 = self.customLayers.fullcon(d1,self.customLayers.get_weight(vshape=[1024,512],
                                                                         name="d2"))
        d3 = self.customLayers.fullcon(d2,self.customLayers.get_weight(vshape=[512,128],
                                                                         name="d3"))
        if(verbose):
            num_features = self.customLayers.num_params
            print("flat",flat.shape)
        
        fc1 = self.customLayers.fullcon(d3,self.customLayers.get_weight(vshape=[128,self.num_classes],
                                                                         name="fullcon1"))
        if(verbose):
            print("fullcon1",fc1.shape)
            print("\nNum Parameters")
            print("Feature Extractor : %d"%(num_features))
            print("Classifier        : %d"%(self.customLayers.num_params - num_features))
            print("Total             : %d"%(self.customLayers.num_params))
        return fc1
    def residual(self,inputs,ksize,inchannel,outchannel, name = "",verbose=False):
        convtmp_1 = self.customLayers.conv2d(inputs,self.customLayers.get_weight(vshape = [ksize,ksize,inchannel,outchannel],
                                                                    name ="%s_1"%(name)),
                                        stride_size =1, padding='SAME')
        convtmp_1bn = self.customLayers.batch_norm(convtmp_1,name="%s_1bn"%(name))
        convtmp_1act = self.customLayers.elu(convtmp_1bn)
        
        convtmp_2 = self.customLayers.conv2d(convtmp_1act,self.customLayers.get_weight(vshape = [ksize,ksize,outchannel,outchannel],
                                                                    name ="%s_2"%(name)),
                                        stride_size =1, padding='SAME')
        convtmp_2bn = self.customLayers.batch_norm(convtmp_2,name="%s_2bn"%(name))
        convtmp_2act = self.customLayers.elu(convtmp_2bn)
        
        if(inputs.shape[-1] != convtmp_2act.shape[-1]):
            convtmp_sc = self.customLayers.conv2d(inputs,
                                                  self.customLayers.get_weight(vshape = [1,1,inchannel,outchannel],
                                                                    name ="%s_sc"%(name)),
                                        stride_size =1, padding='SAME')
            convtmp_scbn = self.customLayers.batch_norm(convtmp_sc,name="%s_scbn"%(name))
            convtmp_scact = self.customLayers.elu(convtmp_scbn)
            inputs = convtmp_scact
        
        output = inputs + convtmp_2act
        if(verbose): print(name,output.shape)
        return output