import numpy as np
import random
import collections
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp
from dice_ml.utils.sample_architecture.vae_model import CF_VAE

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class DiceBaseGenCF:

    def __init__(self, data_interface, model_interface):
        """
        :param data_interface: an interface class to data related params
        :param model_interface: an interface class to access trained ML model
        """    
        
        self.pred_model= model_interface
        
        self.encoded_size=10
        self.data_size = len(data_interface.encoded_feature_names)

        # Dataset for training Variational Encoder Decoder model for CF Generation
        train_data_vae= data_interface.data_df.copy()
        # Can remove the condition of training on only low income dataset
        train_data_vae= train_data_vae[ train_data_vae['income']==0 ]

        #MAD
        self.mad_feature_weights = data_interface.get_mads_from_training_data(normalized=False)

        #One Hot Encoding for categorical features
        encoded_data = data_interface.one_hot_encode_data(train_data_vae)
        dataset = encoded_data.to_numpy()

        #Normlaise_Weights
        self.normalise_weights={}
        encoded_categorical_feature_indexes = data_interface.get_data_params()[2]     
        encoded_continuous_feature_indexes=[]
        for i in range(data_size):
            valid=1
            for v in encoded_categorical_feature_indexes:
                if i in v:
                    valid=0
            if valid:
                encoded_continuous_feature_indexes.append(i)            
        encoded_start_cat = len(encoded_continuous_feature_indexes)
        for idx in encoded_continuous_feature_indexes:
            _max= float(np.max( dataset[:,idx] ))
            _min= float(np.min( dataset[:,idx] ))
            self.normalise_weights[idx]=[_min, _max]

        #Normlization for conitnuous features
        encoded_data= d.normalize_data(encoded_data)
        dataset = encoded_data.to_numpy()

        #Train, Val, Test Splits
        np.random.shuffle(dataset)
        test_size= int(0.1*dataset.shape[0])
        self.vae_test_dataset= dataset[:test_size]
        dataset= dataset[test_size:]
        self.vae_val_dataset= dataset[:test_size]
        self.vae_train_dataset= dataset[test_size:]

        #BaseGenCF Model
        self.cf_vae = CF_VAE(self.data_size, self.encoded_size, data_interface)
        
        #Hyperparam 
        # Currently set to the specific values for the Adult dataset; dataset dependent
        # TODO: Make these hyperparam dataset dependent
        self.learning_rate= 1e-2
        self.batch_size= 2048
        self.validity_reg= 42.0 
        self.margin= 0.165
        self.epoch= 25
       
        #Optimizer    
        self.cf_vae_optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
            {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
            {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3},
            ], lr=learning_rate
        )
    
        def compute_loss( self, model_out, x, target_label ): 

            em = model_out['em']
            ev = model_out['ev']
            z  = model_out['z']
            dm = model_out['x_pred']
            mc_samples = model_out['mc_samples']
            #KL Divergence
            kl_divergence = 0.5*torch.mean( em**2 +ev - torch.log(ev) - 1, axis=1 ) 

            #Reconstruction Term
            #Proximity: L1 Loss
            x_pred = dm[0]       
            s= model.encoded_start_cat
            recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
            for key in self.normalise_weights.keys():
                # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
                recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

            # Sum to 1 over the categorical indexes of a feature
            for v in self.model.encoded_categorical_feature_indexes:
                temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
                recon_err += temp

            count=0
            count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
            count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    

            #Validity         
            temp_logits = self.pred_model(x_pred)
            validity_loss= torch.zeros(1).to(cuda)                                        
            temp_1= temp_logits[target_label==1,:]
            temp_0= temp_logits[target_label==0,:]
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), self.margin, reduction='mean')
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), self.margin, reduction='mean')

            for i in range(1,mc_samples):
                x_pred = dm[i]       

                recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
                for key in self.normalise_weights.keys():
                    # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
                    recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

                # Sum to 1 over the categorical indexes of a feature
                for v in model.encoded_categorical_feature_indexes:
                    temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
                    recon_err += temp

                count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
                count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        

                #Validity
                temp_logits = self.pred_model(x_pred)
        #         validity_loss += -F.cross_entropy(temp_logits, target_label)      
                temp_1= temp_logits[target_label==1,:]
                temp_0= temp_logits[target_label==0,:]
                validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]).to(cuda) - F.sigmoid(temp_1[:,0]).to(cuda), torch.tensor(-1).to(cuda), self.margin, reduction='mean')
                validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]).to(cuda) - F.sigmoid(temp_0[:,1]).to(cuda), torch.tensor(-1).to(cuda), self.margin, reduction='mean')

            recon_err = recon_err / mc_samples
            validity_loss = -1*self.validity_reg*validity_loss/mc_samples

            print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
            print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
            return -torch.mean(recon_err - kl_divergence) - validity_loss

    
    def train( self ):
        
        for epoch in self.epochs:
            batch_num=0
            train_loss= 0.0
            train_size=0
            
            train_dataset= torch.tensor(self.vae_train_dataset).float()
            train_dataset= torch.utils.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_x in enumerate(train_dataset):
                self.cf_vae_optimizer.zero_grad()
                
                train_x= train_x[1]
                train_y= 1.0-torch.argmax( self.pred_model(train_x), dim=1 )
                train_size+= train_x.shape[0]
                
                out= self.cf_vae(train_x, train_y)
                loss= self.compute_loss( out, train_x, train_y )
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
                batch_num+=1
                
            ret= loss/batch_num
            print('Train Avg Loss: ', ret, train_size )
            
    ## I plan to keep the input arguments for this function same as the one defined for Diverse CF
    ## Questions: Would query_instance a numpy_matrix
    ## In what format do I need to return the predicted countefactuals
    
    def generate_countefactuals(self, query_instance, total_CFs, desired_class="opposite" ):
        
        # Converting query_instance into numpy array
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values])
        
        test_dataset= np.array_split( query_instance, query_instance.shape[0]//batch_size ,axis=0 )
        final_gen_cf=[]
        final_cf_pred=[]
        for i in range(len(query_instance)):
            train_x = test_dataset[i]
            train_x= torch.tensor( train_x ).float().to(cuda)
            train_y = torch.argmax( self.pred_model(train_x), dim=1 )                
            train_size += train_x.shape[0]        
            curr_gen_cf=[]
            curr_cf_pred=[]            
            
            for cf_count in range(total_CFs):                
                recon_err, kl_err, x_true, x_pred, cf_label = model.compute_elbo( train_x, 1.0-train_y, pred_model )
                curr_gen_cf.append(x_pred.cpu().numpy())
                curr_cf_pred.append(cf_label.cpu.numpy())
# Code for converting tensor countefactuals into pandas dataframe                
#                 x_pred= d.de_normalize_data( d.get_decoded_data(x_pred.detach().cpu().numpy()) )
#                 x_true= d.de_normalize_data( d.get_decoded_data(x_true.detach().cpu().numpy()) )                
            gen_cf.append(curr_gen_cf)
    
        return gen_cf
        
