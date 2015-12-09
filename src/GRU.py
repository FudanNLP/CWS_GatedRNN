import numpy as np
from MLP import *
from Activation import *

from collections import OrderedDict

class GRU_LayerSetting(object):
    def __init__(self,
                 flag_use_mlps,
                 n_rnn_layer,
                 gate_activation,
                 activation,
                 mlp_layersizes,
                 n_input_unit,
                 wordVecLen):
        self.gate_activation = gate_activation
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
        
        
class GRU_HiddenLayer(object):
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
        
        self.gate_activation = layer_setting.gate_activation
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
        
        
        self.zGate = Layer(rng = rng, n_in = n_hidden * 3,n_out = n_hidden,activation = self.gate_activation)
        self.params['zGate_W'] = self.zGate.W
        self.learning_rate['zGate_W'] = np.ones(self.zGate.W.shape, dtype=np.float32)
        self.batch_grad['zGate_W'] = np.zeros(self.zGate.W.shape, dtype=np.float32)
        
        self.rgGate = Layer(rng = rng, n_in = n_hidden * 3,n_out = n_hidden,activation = self.gate_activation)
        self.params['rgGate_W'] = self.rgGate.W
        self.learning_rate['rgGate_W'] = np.ones(self.rgGate.W.shape, dtype=np.float32)
        self.batch_grad['rgGate_W'] = np.zeros(self.rgGate.W.shape, dtype=np.float32)
        
        self.hidden = Layer(rng = rng, n_in = n_hidden * 2,n_out = n_hidden,activation = self.activation)
        self.params['hidden_W'] = self.hidden.W
        self.learning_rate['hidden_W'] = np.ones(self.hidden.W.shape, dtype=np.float32)
        self.batch_grad['hidden_W'] = np.zeros(self.hidden.W.shape, dtype=np.float32)
        
        self.rLeftGate = Layer(rng = rng, n_in = n_hidden * 2,n_out = n_hidden,activation = self.gate_activation)
        self.params['rLeftGate_W'] = self.rLeftGate.W
        self.learning_rate['rLeftGate_W'] = np.ones(self.rLeftGate.W.shape, dtype=np.float32)
        self.batch_grad['rLeftGate_W'] = np.zeros(self.rLeftGate.W.shape, dtype=np.float32)
        
        self.rRightGate = Layer(rng = rng, n_in = n_hidden * 2,n_out = n_hidden,activation = self.gate_activation)
        self.params['rRightGate_W'] = self.rRightGate.W
        self.learning_rate['rRightGate_W'] = np.ones(self.rRightGate.W.shape, dtype=np.float32)
        self.batch_grad['rRightGate_W'] = np.zeros(self.rRightGate.W.shape, dtype=np.float32)
        
        self.input2hidden = Layer(rng = rng, n_in = n_in / self.n_input_unit,n_out = n_hidden,activation = self.activation) 
        self.params['input2hidden_W'] = self.input2hidden.W
        self.learning_rate['input2hidden_W'] = np.ones(self.input2hidden.W.shape, dtype=np.float32)
        self.batch_grad['input2hidden_W'] = np.zeros(self.input2hidden.W.shape, dtype=np.float32)
        
        if(use_bias == True):
            self.params['zGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['zGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['zGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['rgGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['rgGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['rgGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['rLeftGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['rLeftGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['rLeftGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['rRightGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['rRightGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['rRightGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['hidden_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['hidden_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['hidden_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['input2hidden_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['input2hidden_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['input2hidden_b'] = np.zeros(n_hidden, dtype=np.float32)
        
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
        self.hat_hidden_set = []
        self.zGate_set = []
        self.rgGate_set = []
        self.rLeftGate_set = []
        self.rRightGate_set = []
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
        hat_hidden_set = []
        zGate_set = []
        rgGate_set = []
        rLeftGate_set = []
        rRightGate_set = []
        
        
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
            cur_hat_hidden_list = []
            cur_zGate_list = []
            cur_rgGate_list = []
            cur_rLeftGate_list = []
            cur_rRightGate_list = []
            for hZip in zip(pre_output[:-1],pre_output[1:]):
                (hLeft,hRight) = hZip
                h_pre = np.concatenate([hLeft,hRight])
                #print hLeft.shape,h_pre.shape,self.params['rLeftGate_W'].shape
                inside = np.dot(h_pre,self.params['rLeftGate_W'])
                if(self.use_bias == True): inside += self.params['rLeftGate_b']
                rLeft_gate = self.gate_activation.encode(inside)
                
                inside = np.dot(h_pre,self.params['rRightGate_W'])
                if(self.use_bias == True): inside += self.params['rRightGate_b']
                rRight_gate = self.gate_activation.encode(inside)
                
                inside = np.dot(np.concatenate([hLeft * rLeft_gate, hRight * rRight_gate])
                                ,self.params['hidden_W'])
                if(self.use_bias == True): inside += self.params['hidden_b']
                hat_hidden = self.activation.encode(inside)
                
                inside = np.dot(np.concatenate([h_pre,hat_hidden]),self.params['zGate_W'])
                if(self.use_bias == True): inside += self.params['zGate_b']
                z_gate = self.gate_activation.encode(inside)
                
                inside = np.dot(np.concatenate([h_pre,hat_hidden]),self.params['rgGate_W'])
                if(self.use_bias == True): inside += self.params['rgGate_b']
                rg_gate = self.gate_activation.encode(inside)
                
                cur_hidden = (1 - z_gate) * ((1 - rg_gate) * hLeft + rg_gate * hRight) + \
                                z_gate * hat_hidden
                
                cur_hidden_list.append(cur_hidden)
                cur_hat_hidden_list.append(hat_hidden)
                cur_zGate_list.append(z_gate)
                cur_rgGate_list.append(rg_gate)
                cur_rLeftGate_list.append(rLeft_gate)
                cur_rRightGate_list.append(rRight_gate)
            '''
            hidden_set.append(cur_hidden_list)
            hat_hidden_set.append(cur_hat_hidden_list)
            zGate_set.append(cur_zGate_list)
            rgGate_set.append(cur_rgGate_list)
            rLeftGate_set.append(cur_rLeftGate_list)
            rRightGate_set.append(cur_rRightGate_list)
            pre_output = cur_hidden_list
            '''
            hidden_set += cur_hidden_list
            hat_hidden_set += cur_hat_hidden_list
            zGate_set += cur_zGate_list
            rgGate_set += cur_rgGate_list
            rLeftGate_set += cur_rLeftGate_list
            rRightGate_set += cur_rRightGate_list
            
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
            self.hat_hidden_set.append(hat_hidden_set)
            self.zGate_set.append(zGate_set)
            self.rgGate_set.append(rgGate_set)
            self.rLeftGate_set.append(rLeftGate_set)
            self.rRightGate_set.append(rRightGate_set)
            self.input.append(x_in)
        

        return output
    
    def get_output_mlp(self,mlp,x_in,flag):
        for layer in mlp:
            x_in = layer.encode(x_in,flag)
        return x_in
    
    def get_pos(self,pos):
        pos_hidden = pos
        pos_hat_hidden = pos - self.n_input_unit
        pos_zGate = pos - self.n_input_unit
        pos_rgGate = pos - self.n_input_unit
        pos_rLeftGate = pos - self.n_input_unit
        pos_rRightGate = pos - self.n_input_unit
        pos_leftHidden = self.table_father2son[pos]
        pos_rightHidden = self.table_father2son[pos] + 1
        return (pos_hidden,pos_hat_hidden,pos_zGate,
                pos_rgGate,pos_rLeftGate,pos_rRightGate,
                pos_leftHidden,pos_rightHidden)
    
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
        
        
        hat_hidden = []
        zGate = []
        rgGate = []
        rLeftGate = []
        rRightGate = []
        for i in xrange(count_unit - self.n_input_unit):
            hat_hidden_list = []
            zGate_list = []
            rgGate_list = []
            rLeftGate_list = []
            rRightGate_list = []
            for pos in xrange(l_sen):
                hat_hidden_list.append(self.hat_hidden_set[pos][i])
                zGate_list.append(self.zGate_set[pos][i])
                rgGate_list.append(self.rgGate_set[pos][i])
                rLeftGate_list.append(self.rLeftGate_set[pos][i])
                rRightGate_list.append(self.rRightGate_set[pos][i])
            hat_hidden_list = np.asarray(hat_hidden_list)
            zGate_list = np.asarray(zGate_list)
            rgGate_list = np.asarray(rgGate_list)
            rLeftGate_list = np.asarray(rLeftGate_list)
            rRightGate_list = np.asarray(rRightGate_list)
            hat_hidden.append(hat_hidden_list)
            zGate.append(zGate_list)
            rgGate.append(rgGate_list)
            rLeftGate.append(rLeftGate_list)
            rRightGate.append(rRightGate_list)
        
        
        
        width = self.layer_setting.wordVecLen
        g_ = {}
        for param in self.params:
            g_[param] = np.zeros(self.params[param].shape,dtype = np.float32)
            
        
        for pos in reversed(xrange(self.n_input_unit,count_unit)):
            (pos_hidden,pos_hat_hidden,pos_zGate,
                pos_rgGate,pos_rLeftGate,pos_rRightGate,
                pos_leftHidden,pos_rightHidden) = self.get_pos(pos)
            #print self.get_pos(pos)
            #print len(hidden),pos_hidden
            cur_hidden = hidden[pos_hidden]
            cur_hat_hidden = hat_hidden[pos_hat_hidden]
            cur_zGate = zGate[pos_zGate]
            cur_rgGate = rgGate[pos_rgGate]
            cur_rLeftGate = rLeftGate[pos_rLeftGate]
            cur_rRightGate = rRightGate[pos_rRightGate]
            cur_leftHidden = hidden[pos_leftHidden]
            cur_rightHidden = hidden[pos_rightHidden]
            
            #print cur_zGate.shape
            g_cur_hidden = g_hidden[pos]
            g_cur_hat_hidden = g_cur_hidden * cur_zGate
            g_cur_leftHidden = g_cur_hidden * (1 - cur_zGate) * (1 - cur_rgGate)
            g_cur_rightHidden = g_cur_hidden * (1 - cur_zGate) * cur_rgGate
            g_cur_zGate = g_cur_hidden * (cur_hat_hidden - (1 - cur_rgGate) * cur_leftHidden - cur_rgGate * cur_rightHidden)
            g_cur_rgGate = g_cur_hidden * (1 - cur_zGate) * (cur_rightHidden - cur_leftHidden)
            
            tmp = g_cur_zGate * self.gate_activation.bp(cur_zGate)
            g_['zGate_W'] += np.concatenate([cur_leftHidden,cur_rightHidden,cur_hat_hidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['zGate_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['zGate_W'].T)
            g_cur_leftHidden += tmp_g[:,:width]
            g_cur_rightHidden += tmp_g[:,width:2*width]
            g_cur_hat_hidden += tmp_g[:,2*width:]
            
            tmp = g_cur_rgGate * self.gate_activation.bp(cur_rgGate)
            g_['rgGate_W'] += np.concatenate([cur_leftHidden,cur_rightHidden,cur_hat_hidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['rgGate_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['rgGate_W'].T)
            g_cur_leftHidden += tmp_g[:,:width]
            g_cur_rightHidden += tmp_g[:,width:2*width]
            g_cur_hat_hidden += tmp_g[:,2*width:]
            
            tmp = g_cur_hat_hidden * self.activation.bp(cur_hat_hidden)
            g_['hidden_W'] += np.concatenate([cur_rLeftGate * cur_leftHidden,
                                             cur_rRightGate * cur_rightHidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['hidden_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['hidden_W'].T)
            g_cur_leftHidden += tmp_g[:,:width] * cur_rLeftGate
            g_cur_rightHidden += tmp_g[:,width:] * cur_rRightGate
            g_cur_rLeftGate = tmp_g[:,:width] * cur_leftHidden
            g_cur_rRightGate = tmp_g[:,width:] * cur_rightHidden
            
            tmp = g_cur_rLeftGate * self.gate_activation.bp(cur_rLeftGate)
            g_['rLeftGate_W'] += np.concatenate([cur_leftHidden,cur_rightHidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['rLeftGate_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['rLeftGate_W'].T)
            g_cur_leftHidden += tmp_g[:,:width]
            g_cur_rightHidden += tmp_g[:,width:]
            
            tmp = g_cur_rRightGate * self.gate_activation.bp(cur_rRightGate)
            g_['rRightGate_W'] += np.concatenate([cur_leftHidden,cur_rightHidden],axis = 1).T.dot(tmp)
            if(self.use_bias == True): g_['rRightGate_b'] += np.sum(tmp, axis = 0)
            tmp_g = tmp.dot(self.params['rRightGate_W'].T)
            g_cur_leftHidden += tmp_g[:,:width]
            g_cur_rightHidden += tmp_g[:,width:]
            
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
        self.hat_hidden_set = []
        self.zGate_set = []
        self.rgGate_set = []
        self.rLeftGate_set = []
        self.rRightGate_set = []
        self.input = []
        
        if(self.flag_use_mlps == True):
            for mlp in self.mlps:
                for layer in mlp:
                    layer.clear_layers()
        


        
        