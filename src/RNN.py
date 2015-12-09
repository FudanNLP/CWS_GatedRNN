import numpy as np
from MLP import *
from Activation import *

from collections import OrderedDict

class RNN_LayerSetting(object):
    def __init__(self,
                 flag_use_mlps,
                 n_rnn_layer,
                 
                 activation,
                 mlp_layersizes,
                 n_input_unit,
                 wordVecLen):
        #self.gate_activation = gate_activation
        self.activation = activation
        self.mlp_layersizes=mlp_layersizes
        self.n_input_unit = n_input_unit
        self.flag_use_mlps=flag_use_mlps
        self.n_rnn_layer=n_rnn_layer
        self.wordVecLen=wordVecLen
        
        cnt = 0
        n = 0
        for i in reversed(xrange(self.n_input_unit)):
            n += i + 1
            cnt += 1
            if(cnt >= self.n_rnn_layer): break
        self.n_unit = n
        self.outputSize = wordVecLen * n 
        if(self.flag_use_mlps == True):
            self.outputSize = self.mlp_layersizes[-1] * n
        
        
class RNN_HiddenLayer(object):
    def __init__(self,alpha,
                     squared_filter_length_limit,
                     L2_reg,
                     flag_dropout,
                     n_in, n_out, use_bias,
                     dropout_rate,
                     flag_dropout_scaleWeight,
                     layer_setting,
                     rng):
        #print n_in,n_out
        self.flag_dropout = flag_dropout
        self.alpha = alpha
        self.squared_filter_length_limit = squared_filter_length_limit
        self.L2_reg = L2_reg
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self.rng = rng
        self.dropout_rate = dropout_rate
        self.flag_dropout_scaleWeight = flag_dropout_scaleWeight
        
        #self.gate_activation = layer_setting.gate_activation
        self.activation = layer_setting.activation
        self.n_input_unit = layer_setting.n_input_unit
        self.n_rnn_layer = layer_setting.n_rnn_layer
        self.flag_use_mlps = layer_setting.flag_use_mlps
        self.n_unit = layer_setting.n_unit
        self.layer_setting = layer_setting
        
        n_hidden = layer_setting.wordVecLen
        self.params = OrderedDict({})
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})
        
        
        
        self.hiddenLayer = Layer(rng = rng, n_in = n_hidden * 2,n_out = n_hidden,activation = self.activation)
        self.params['hiddenLayer_W'] = self.hiddenLayer.W
        self.learning_rate['hiddenLayer_W'] = np.ones(self.hiddenLayer.W.shape, dtype=np.float32)
        self.batch_grad['hiddenLayer_W'] = np.zeros(self.hiddenLayer.W.shape, dtype=np.float32)
        
        
        
        if(use_bias == True):
            self.params['hiddenLayer_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['hiddenLayer_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['hiddenLayer_b'] = np.zeros(n_hidden, dtype=np.float32)
            
    
        
        if(self.layer_setting.flag_use_mlps == True):
            self.mlps = []
            mlp_layerSetting = MLP_LayerSetting(activation = Tanh())
            mlp_layersizes = self.layer_setting.mlp_layersizes
            weight_matrix_sizes = zip(mlp_layersizes[:-1], mlp_layersizes[1:])
            
            cnt_rnn_layer = 0
            for i in reversed(range(self.n_input_unit)):
                if(cnt_rnn_layer >= self.n_rnn_layer):break
                cnt_rnn_layer += 1
                for j in xrange(i+1):
                    mlp_layers = []      
                    for n_in, n_out in weight_matrix_sizes:
                        mlp_layers.append(MLP_HiddenLayer(alpha=alpha,
                                                             squared_filter_length_limit=squared_filter_length_limit,
                                                             L2_reg = L2_reg,
                                                             flag_dropout=False,
                                                             n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                             dropout_rate=0.0,
                                                             flag_dropout_scaleWeight=flag_dropout_scaleWeight,
                                                             layer_setting = mlp_layerSetting,
                                                             rng=self.rng))
                    self.mlps.append(mlp_layers)
        
        
        self.mask = []
        
        
        self.output = []
        self.hidden_set = []
        self.input = []
        
        
        self.table_father2son = {}
        
        cnt = 0
        for i in reversed(xrange(1,self.n_input_unit)):
            for j in xrange(i+1):
                self.table_father2son[cnt+j+i+1] = cnt+j
            cnt += (i + 1)
                
    
    def _mask_maker(self,x_in):
        mask = self.rng.binomial(n=1, p=1-self.dropout_rate, size=x_in.shape)
        return mask
    
    def encode(self,x_in,flag_train_phase):
        w_scale = 1.0
        if(self.flag_dropout == True) and (flag_train_phase == True):
            mask = self._mask_maker(x_in)
            x_in = x_in * mask
            self.mask.append(mask)
        elif (self.flag_dropout == True) and (flag_train_phase == False):
            w_scale = 1.0-self.dropout_rate
        if(self.flag_dropout_scaleWeight == False): w_scale = 1.0
        
        hidden_set = []
        
        
        
        tmp_len = x_in.shape[0] / self.n_input_unit
        pre_output = []
        for i in xrange(self.n_input_unit):
            pre_output.append(x_in[i*tmp_len:(i+1)*tmp_len])
        hidden_set += pre_output
        
        cnt_rnn_layer = 1
        for layer in xrange(self.n_input_unit-1):
            if(cnt_rnn_layer >= self.n_rnn_layer): break
            cnt_rnn_layer += 1
            cur_hidden_list = []
            for hZip in zip(pre_output[:-1],pre_output[1:]):
                (hLeft,hRight) = hZip
                h_pre = np.concatenate([hLeft,hRight])
                #print hLeft.shape,h_pre.shape
                inside = np.dot(h_pre,self.params['hiddenLayer_W'])
                if(self.use_bias == True): inside += self.params['hiddenLayer_b']
                cur_hidden = self.activation.encode(inside)
                
                
                cur_hidden_list.append(cur_hidden)
            
            hidden_set += cur_hidden_list
            
            pre_output = cur_hidden_list
        
        #self.output = []
        
        if(self.flag_use_mlps == True):
            count_mlp = 0
            output = np.asarray([])
            for unit in hidden_set:
                cur_mlp = self.mlps[count_mlp]
                output_cur_mlp = self.get_output_mlp(cur_mlp,unit,flag_train_phase)
                output = np.concatenate([output,output_cur_mlp])
                count_mlp += 1
        else:
            output = np.asarray([])
            for unit in hidden_set:
                output = np.concatenate([output,unit])
                
                
        if(flag_train_phase == True):
            self.output.append(output)
            self.hidden_set.append(hidden_set)
            self.input.append(x_in)
        

        return output
    
    def get_output_mlp(self,mlp,x_in,flag):
        for layer in mlp:
            x_in = layer.encode(x_in,flag)
        return x_in
    
    def get_pos(self,pos):
        pos_hidden = pos
        pos_leftHidden = self.table_father2son[pos]
        pos_rightHidden = self.table_father2son[pos] + 1
        return (pos_hidden,pos_leftHidden,pos_rightHidden)
    
    def get_gradient(self,g_uplayer,cmd):
        l_sen = g_uplayer.shape[0]
        count_unit = self.n_unit
        if(self.flag_use_mlps == True):
            g_mlps = []
            n_mlps = len(self.mlps)
            width = self.layer_setting.mlp_layersizes[-1]
            for i in xrange(n_mlps):
                g_mlps.append(g_uplayer[:,i*width:(i+1)*width])
            for i in xrange(n_mlps):
                for layer in reversed(self.mlps[i]):
                    g_mlps[i] = layer.get_gradient(g_mlps[i],cmd)
            g_hidden = g_mlps
        else:
            g_hidden = []
            width = self.layer_setting.wordVecLen
            for i in xrange(count_unit):
                g_hidden.append(g_uplayer[:,i*width:(i+1)*width])
        
        
        hidden = []
        for i in xrange(count_unit):
            hidden_list = []
            for pos in xrange(l_sen):
                hidden_list.append(self.hidden_set[pos][i])
            hidden_list = np.asarray(hidden_list)
            hidden.append(hidden_list)
        
        width = self.layer_setting.wordVecLen
        g_ = {}
        for param in self.params:
            g_[param] = np.zeros(self.params[param].shape,dtype = np.float32)
            
        
        for pos in reversed(xrange(self.n_input_unit,count_unit)):
            (pos_hidden,pos_leftHidden,pos_rightHidden) = self.get_pos(pos)
            #print self.get_pos(pos)
            #print len(hidden),pos_hidden
            cur_hidden = hidden[pos_hidden]
            cur_leftHidden = hidden[pos_leftHidden]
            cur_rightHidden = hidden[pos_rightHidden]
            
            #print cur_zGate.shape
            g_cur_hidden = g_hidden[pos]
            
            tmp = g_cur_hidden * self.activation.bp(cur_hidden)
            g_['hiddenLayer_W'] += np.concatenate([cur_leftHidden,cur_rightHidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['hiddenLayer_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['hiddenLayer_W'].T)
            g_cur_leftHidden = tmp_g[:,:width]
            g_cur_rightHidden = tmp_g[:,width:]
            
            
            g_hidden[pos_leftHidden] += g_cur_leftHidden
            g_hidden[pos_rightHidden] += g_cur_rightHidden
        
        
        
                
        g_z = g_hidden[0]
        for i in xrange(1,self.n_input_unit):
            g_z = np.concatenate([g_z,g_hidden[i]],axis = 1)
        
        if(cmd == 'minus'):
            for param in g_:
                g_[param] = -g_[param]
        self.g_ = g_
        
        
        for param in g_:
            self.batch_grad[param] += g_[param]
        
        #print g_['hiddenLayer_W']
        #print g_z
        
        if(self.flag_dropout == True):
            #mask = np.asarray(self.mask,dtype = np.float32)
            mask = np.asarray(self.mask[:])
            return g_z * mask
        return g_z
    
    
    def _scale(self,param,squared_filter_length_limit):
        if(squared_filter_length_limit == False):return param
        col_norms = np.sqrt(np.sum(param**2, axis=0))
        desired_norms = np.clip(col_norms, 0, np.sqrt(squared_filter_length_limit))
        scale = desired_norms / (1e-7 + col_norms)
        return param*scale
    
    
    def update_w(self,n):
        if(self.flag_use_mlps == True):
            for mlp in self.mlps:
                for layer in mlp:
                    layer.update_w(n)
        for param in self.params:
            old_param = self.params[param]
            if(old_param.ndim == 2):
                grad = self.batch_grad[param] / n + self.L2_reg * old_param
            else:
                grad = self.batch_grad[param] / n
            tmp = self.learning_rate[param] + grad * grad
            lr = self.alpha / (np.sqrt(tmp) + 1.0e-6)
            if(old_param.ndim == 2):
                self.params[param] = self._scale(old_param - lr * grad,self.squared_filter_length_limit)
            else:
                self.params[param] = old_param - lr * grad
            self.learning_rate[param] = tmp
            self.batch_grad[param] = np.zeros_like(old_param,dtype = np.float32)
        #clear
        self.clear_layers()
    def clear_layers(self):
        self.mask = []
        
        self.output = []
        self.hidden_set = []
        self.input = []
        
        if(self.flag_use_mlps == True):
            for mlp in self.mlps:
                for layer in mlp:
                    layer.clear_layers()
        

        