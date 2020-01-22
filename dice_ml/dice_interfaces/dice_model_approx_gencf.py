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

class DiceModelApproxGenCF:

    def __init__(self, data_interface, model_interface):
        """
        :param data_interface: an interface class to data related params
        :param model_interface: an interface class to access trained ML model
        """    
        
        self.pred_model= model_interface
        self.data_interface= data_interface
        
        self.encoded_size=10
        self.data_size = len(self.data_interface.encoded_feature_names)

        # Dataset for training Variational Encoder Decoder model for CF Generation
        train_data_vae= self.data_interface.data_df.copy()
        
        #MAD
        #self.mad_feature_weights = self.data_interface.get_mads_from_training_data(normalized=False)

        #Creating list of encoded categorical and continuous feature indices
        encoded_categorical_feature_indexes = self.data_interface.get_data_params()[2]     
        encoded_continuous_feature_indexes=[]
        for i in range(self.data_size):
            valid=1
            for v in encoded_categorical_feature_indexes:
                if i in v:
                    valid=0
            if valid:
                encoded_continuous_feature_indexes.append(i)            
        encoded_start_cat = len(encoded_continuous_feature_indexes)
        
        #One Hot Encoding for categorical features
        encoded_data = self.data_interface.one_hot_encode_data(train_data_vae)
        
        # The output/outcome variable position altered due to one_hot_encoding for categorical features: (Cont feat, Outcome, Cat feat) 
        # Need to rearrange columns such that outcome variable comes at the last
        cols = list(encoded_data.columns)
        cols = cols[:encoded_start_cat] + cols[encoded_start_cat+1:] + [cols[encoded_start_cat]]
        encoded_data = encoded_data[cols]        

        #Normlaise_Weights
        self.normalise_weights={}        
        dataset = encoded_data.to_numpy()
        for idx in encoded_continuous_feature_indexes:
            _max= float(np.max( dataset[:,idx] ))
            _min= float(np.min( dataset[:,idx] ))
            self.normalise_weights[idx]=[_min, _max]

        # Normlization for continuous features
        encoded_data= self.data_interface.normalize_data(encoded_data)
        dataset = encoded_data.to_numpy()

        #Train, Val, Test Splits
        np.random.shuffle(dataset)
        test_size= int(0.1*dataset.shape[0])
        self.vae_test_dataset= dataset[:test_size]
        dataset= dataset[test_size:]
        self.vae_val_dataset= dataset[:test_size]
        self.vae_train_dataset= dataset[test_size:]

        #BaseGenCF Model
        self.cf_vae = CF_VAE(self.data_size, self.encoded_size, self.data_interface)
        
        #Hyperparam 
        # Currently set to the specific values for the Adult dataset; dataset dependent
        # TODO: Make these hyperparam dataset dependent
        self.learning_rate= 1e-2
        self.batch_size= 2048
        self.validity_reg= 42.0 
        self.margin= 0.165
        self.epochs= 25
        self.wm1=1e-2
        self.wm2=1e-2
        self.wm3=1e-2
       
        #Optimizer    
        self.cf_vae_optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, self.cf_vae.encoder_mean.parameters()),'weight_decay': self.wm1},
            {'params': filter(lambda p: p.requires_grad, self.cf_vae.encoder_var.parameters()),'weight_decay': self.wm2},
            {'params': filter(lambda p: p.requires_grad, self.cf_vae.decoder_mean.parameters()),'weight_decay': self.wm3},
            ], lr=self.learning_rate
        )
        
        self.base_model_dir= '../dice_ml/utils/sample_trained_models/'
        self.dataset_name= 'adult'
        ##TODO: A general method to identify the dataset_name
        self.save_path=self.base_model_dir+ self.dataset_name +'-margin-' + str(self.margin) + '-validity_reg-'+ str(self.validity_reg) + '-epoch-' + str(self.epochs) + '-' + 'base-gen' + '.pth'
    
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
        s= self.cf_vae.encoded_start_cat
        recon_err = -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
        for key in self.normalise_weights.keys():
            # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 
            recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

        # Sum to 1 over the categorical indexes of a feature
        for v in self.cf_vae.encoded_categorical_feature_indexes:
            temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
            recon_err += temp

        count=0
        count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
        count+= torch.sum(x_pred[:,:s]>1,axis=1).float()    

        #Validity         
        temp_logits = self.pred_model(x_pred)
        validity_loss= torch.zeros(1)                                       
        temp_1= temp_logits[target_label==1,:]
        temp_0= temp_logits[target_label==0,:]
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]) - F.sigmoid(temp_1[:,0]), torch.tensor(-1), self.margin, reduction='mean')
        validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]) - F.sigmoid(temp_0[:,1]), torch.tensor(-1), self.margin, reduction='mean')

        for i in range(1,mc_samples):
            x_pred = dm[i]       

            recon_err += -torch.sum( torch.abs(x[:,s:-1] - x_pred[:,s:-1]), axis=1 )
            for key in self.normalise_weights.keys():
                # recon_err+= -(1/mad_feature_weights[d.encoded_feature_names[int(key)]])*(normalise_weights[key][1] - normalise_weights[key][0])*torch.abs( (x[:,key] - x_pred[:,key]))
                recon_err+= -(self.normalise_weights[key][1] - self.normalise_weights[key][0])*torch.abs(x[:,key] - x_pred[:,key]) 

            # Sum to 1 over the categorical indexes of a feature
            for v in self.cf_vae.encoded_categorical_feature_indexes:
                temp = -torch.abs(  1.0-torch.sum( x_pred[:, v[0]:v[-1]+1], axis=1) )
                recon_err += temp

            count+= torch.sum(x_pred[:,:s]<0,axis=1).float()
            count+= torch.sum(x_pred[:,:s]>1,axis=1).float()        

            #Validity
            temp_logits = self.pred_model(x_pred)
    #         validity_loss += -F.cross_entropy(temp_logits, target_label)      
            temp_1= temp_logits[target_label==1,:]
            temp_0= temp_logits[target_label==0,:]
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_1[:,1]) - F.sigmoid(temp_1[:,0]), torch.tensor(-1), self.margin, reduction='mean')
            validity_loss += F.hinge_embedding_loss( F.sigmoid(temp_0[:,0]) - F.sigmoid(temp_0[:,1]), torch.tensor(-1), self.margin, reduction='mean')

        recon_err = recon_err / mc_samples
        validity_loss = -1*self.validity_reg*validity_loss/mc_samples

        print('Avg wrong cont dim: ', torch.mean(count)/mc_samples)
        print('recon: ',-torch.mean(recon_err), ' KL: ', torch.mean(kl_divergence), ' Validity: ', -validity_loss)
        return -torch.mean(recon_err - kl_divergence) - validity_loss    
    
    
    def train( self, constraint_type, constraint_variables, constraint_direction, constraint_reg, pre_trained=False  ):
        
        '''        
        pre_trained: Bool Variable to check whether pre trained model exists to avoid training again        
        constraint_type: Binary Variable currently: (1) unary / (0) monotonic
        constraint_variables: List of List: [[Effect, Cause1, Cause2, .... ]]
        constraint_direction: -1: Negative, 1: Positive ( By default has to be one for monotonic constraints )
        constraint_reg: Tunable Hyperparamter
        '''
        
        if pre_trained:
            self.cf_vae.load_state_dict(torch.load(self.save_path))
            self.cf_vae.eval()
            return 
        
        ##TODO: Handling such dataset specific constraints in a more general way
        # CF Generation for only low to high income data points
        self.vae_train_dataset= self.vae_train_dataset[self.vae_train_dataset[:,-1]==0,:]
        self.vae_val_dataset= self.vae_val_dataset[self.vae_val_dataset[:,-1]==0,:]

        #Removing the outcome variable from the datasets
        self.vae_train_feat= self.vae_train_dataset[:,:-1]
        self.vae_val_feat= self.vae_val_dataset[:,:-1]        
        
        for epoch in range(self.epochs):
            batch_num=0
            train_loss= 0.0
            train_size=0
            
            train_dataset= torch.tensor(self.vae_train_feat).float()
            train_dataset= torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for train_x in enumerate(train_dataset):
                self.cf_vae_optimizer.zero_grad()
                
                train_x= train_x[1]
                train_y= 1.0-torch.argmax( self.pred_model(train_x), dim=1 )
                train_size+= train_x.shape[0]
                
                out= self.cf_vae(train_x, train_y)
                loss= self.compute_loss( out, train_x, train_y )
                
                #Unary Case
                if constraint_type:
                    for const in constraint_variables:
                        # Get the index from the feature name
                        # Handle the categorical variable case here too
                        const_idx= const[0]
                        dm = out['x_pred']
                        mc_samples = out['mc_samples']
                        x_pred = dm[0]
                       
                        constraint_loss = F.hinge_embedding_loss( constraint_direction*(x_pred[:,const_idx] - train_x[:,const_idx]), torch.tensor(-1), 0)

                        for j in range(1, mc_samples):
                            x_pred = dm[j]            
                            constraint_loss+= F.hinge_embedding_loss( constraint_direction*(x_pred[:,const_idx] - train_x[:,const_idx]), torch.tensor(-1), 0)           
                            
                        constraint_loss= constraint_loss/mc_samples
                        constraint_loss= constraint_reg*constraint_loss
                        loss+= constraint_loss
                        print('Constraint: ', constraint_loss, torch.mean(constraint_loss) )
                else:
                    #Train the regression model
                    print('Yet to implement')
                    
                loss.backward()
                train_loss += loss.item()
                self.cf_vae_optimizer.step()
                
                batch_num+=1
                
            ret= loss/batch_num
            print('Train Avg Loss: ', ret, train_size )
            
            #Save the model after training every 10 epochs and at the last epoch
            if (epoch!=0 and epoch%10==0) or epoch==self.epochs-1:
                torch.save(self.cf_vae.state_dict(), self.save_path)                      
            
            
    #The input arguments for this function same as the one defined for Diverse CF    
    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite",  ):

        ## Loading the latest trained CFVAE model
        self.cf_vae.load_state_dict(torch.load(self.save_path))
        self.cf_vae.eval()
        
        # Converting query_instance into numpy array
        query_instance_org= query_instance
        
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values])
        
        print(query_instance.shape[0])
        if  query_instance.shape[0] > self.batch_size:
            test_dataset= np.array_split( query_instance, query_instance.shape[0]//self.batch_size ,axis=0 )
        else:
            test_dataset= [ query_instance ] 
        final_gen_cf=[]
        final_cf_pred=[]
        final_test_pred=[]
        for i in range(len(query_instance)):
            train_x = test_dataset[i]
            train_x= torch.tensor( train_x ).float()
            train_y = torch.argmax( self.pred_model(train_x), dim=1 )                
            
            curr_gen_cf=[]
            curr_cf_pred=[]            
            curr_test_pred= train_y.numpy()
            
            for cf_count in range(total_CFs):                                
                recon_err, kl_err, x_true, x_pred, cf_label = self.cf_vae.compute_elbo( train_x, 1.0-train_y, self.pred_model )
                while( cf_label== train_y):
                    print(cf_label, train_y)
                    recon_err, kl_err, x_true, x_pred, cf_label = self.cf_vae.compute_elbo( train_x, 1.0-train_y, self.pred_model )
                    
                x_pred= x_pred.detach().numpy()
                #Converting mixed scores into one hot feature representations
                for v in self.cf_vae.encoded_categorical_feature_indexes:
                    curr_max= x_pred[:, v[0]]
                    curr_max_idx= v[0]
                    for idx in v:
                        if curr_max < x_pred[:, idx]:
                            curr_max= x_pred[:, idx]
                            curr_max_idx= idx
                    for idx in v:
                        if idx==curr_max_idx:
                            x_pred[:, idx]=1
                        else:
                            x_pred[:, idx]=0
                        
                    
                cf_label= cf_label.detach().numpy()
                cf_label= np.reshape( cf_label, (cf_label.shape[0],1) )
                
                curr_gen_cf.append( x_pred )
                curr_cf_pred.append( cf_label )
                
            final_gen_cf.append(curr_gen_cf)
            final_cf_pred.append(curr_cf_pred)
            final_test_pred.append(curr_test_pred)
        
        #CF Gen out
        result={}
        result['query-instance']= query_instance[0]
        result['test-pred']= final_test_pred[0][0]
        result['CF']= final_gen_cf[0]
        result['CF-Pred']= final_cf_pred[0]
        
        # Adding empty list for sparse cf gen and pred; adding 0 for the sparsity coffecient
        return exp.CounterfactualExamples(self.data_interface, result['query-instance'], result['test-pred'], result['CF'], result['CF-Pred'], None, None, 0)