#Common Imports
import numpy as np
import random
import collections
import timeit
import copy

#Dice Imports
from dice_ml.dice_interfaces.dice_base import DiceBase
from dice_ml.dice_interfaces.dice_base_gencf import DiceBaseGenCF
from dice_ml import diverse_counterfactuals as exp
from dice_ml.utils.sample_architecture.vae_model import CF_VAE
from dice_ml.utils.helpers import get_base_gen_cf_initialization

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class DiceModelApproxGenCF(DiceBaseGenCF, DiceBase):

    def __init__(self, data_interface, model_interface):
        """
        :param data_interface: an interface class to data related params
        :param model_interface: an interface class to access trained ML model
        """    
        
        # initiating data related parameters
        DiceBase.__init__(self, data_interface) 
        
        #Black Box ML Model to be explained
        self.pred_model= model_interface.model        
        
        #Hyperparam 
        # Currently set to the specific values for the Adult dataset; dataset dependent
        # TODO: Make these hyperparam dataset dependent
        self.encoded_size=10
        self.learning_rate= 1e-2
        self.batch_size= 2048
        self.validity_reg= 76.0 
        self.margin= 0.344
        self.epochs= 25
        self.wm1=1e-2
        self.wm2=1e-2
        self.wm3=1e-2
       
        #Initializing parameters for the DiceBaseGenCF  
        self.vae_train_dataset, self.vae_val_dataset, self.vae_test_dataset, self.normalise_weights, self.cf_vae, self.cf_vae_optimizer= get_base_gen_cf_initialization( self.data_interface, self.encoded_size, self.cont_minx, self.cont_maxx, self.margin, self.validity_reg, self.epochs, self.wm1, self.wm2, self.wm3, self.learning_rate ) 
        
        #Data paths
        self.base_model_dir= '../dice_ml/utils/sample_trained_models/'
        self.dataset_name= 'adult'
        ##TODO: A general method to identify the dataset_name
        self.save_path=self.base_model_dir+ self.dataset_name +'-margin-' + str(self.margin) + '-validity_reg-'+ str(self.validity_reg) + '-epoch-' + str(self.epochs) + '-' + 'ae-gen' + '.pth'
    
    
    def train( self, constraint_type, constraint_variables, constraint_direction, constraint_reg, pre_trained=False  ):
        
        '''        
        
        :param pre_trained: Bool Variable to check whether pre trained model exists to avoid training again        
        :param constraint_type: Binary Variable currently: (1) unary / (0) monotonic
        :param constraint_variables: List of List: [[Effect, Cause1, Cause2, .... ]]
        :param constraint_direction: -1: Negative, 1: Positive ( By default has to be one for monotonic constraints )
        :param constraint_reg: Tunable Hyperparamter
        
        :return None
        
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
                    raise NotImplementedError("This has not been implemented yet. If you'd like this to be implemented in the next version, please raise an issue at https://github.com/interpretml/DiCE/issues")
                    
                loss.backward()
                train_loss += loss.item()
                self.cf_vae_optimizer.step()
                
                batch_num+=1
                
            ret= loss/batch_num
            print('Train Avg Loss: ', ret, train_size )
            
            #Save the model after training every 10 epochs and at the last epoch
            if (epoch!=0 and epoch%10==0) or epoch==self.epochs-1:
                torch.save(self.cf_vae.state_dict(), self.save_path)