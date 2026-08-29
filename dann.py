import torch
import torch.nn as nn
import math
from einops import rearrange
from torch.autograd import Function
import torch.nn.functional as F


class position_encoding(nn.Module):

    def __init__(self, feature_dim, dropout=0.1, max_len=75):
        super(position_encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape (1, max_len, feature_dim)
        pe = torch.zeros(1, max_len, feature_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)

        # Compute div_term for even indices
        even_indices = torch.arange(0, feature_dim, 2)
        div_term_even = torch.exp(even_indices.float() * (-math.log(10000.0) / feature_dim))
        # Compute positional encodings for even indices
        pe[0, :, 0::2] = torch.sin(position * div_term_even)

        # Compute div_term for odd indices
        odd_indices = torch.arange(1, feature_dim, 2)
        div_term_odd = torch.exp(odd_indices.float() * (-math.log(10000.0) / feature_dim))

       
        # Compute positional encodings for odd indices
        pe[0, :, 1::2] = torch.cos(position * div_term_odd)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe  # Broadcasted addition of positional encoding
        return  self.dropout(x)


class feed_forward(nn.Module):
  def __init__(self,embed_dim ,expansion, drop_p, temporal=False, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    if temporal:
        print('temporal feed')
        self.seq_layer = nn.Sequential(nn.Linear(in_features = embed_dim, out_features = expansion * embed_dim, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(in_features = expansion * embed_dim, out_features = embed_dim, bias=True ))

    else:
        print('spatial  feed')
        self.seq_layer = nn.Sequential(nn.Linear(in_features = embed_dim, out_features = expansion * embed_dim, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(in_features = expansion * embed_dim, out_features = embed_dim, bias=True ))
  
  def forward(self, x):
    return self.seq_layer(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion=2, drop_p=0.1, temporal=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.position_encoding_layer = position_encoding(feature_dim=embed_dim, max_len=74, dropout=drop_p)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_p, batch_first=True)
        self.feed_forward_layer = feed_forward(embed_dim, expansion=expansion, drop_p=drop_p, temporal = temporal)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_p)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, windowsToUse, embed_dim),
                                               requires_grad=True)
        
    def forward(self, x, key_padding_mask, temporal=False ):

        #print( torch.sum(key_padding_mask) )
        if temporal:
            #x = self.position_encoding_layer(x)  # x shape: (batch_size, channels, features)
            x = self.position_embedding + x
            attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask = key_padding_mask ) 
            
            attn_output = attn_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            

        else:
            attn_output, _ = self.multihead_attn(x, x, x  )  # attn_output shape: (batch_size, channels,embed_dim)


        # Add & Norm
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        if temporal:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Feed-forward
        ff_output = self.feed_forward_layer(x)  # ff_output shape: (batch_size, channels, embed_dim)
        if temporal:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Add & Norm
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        if temporal:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x
   

class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, temporal=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a list of TransformerEncoder layers
        self.layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, expansion=expansion, drop_p=drop_p, temporal = temporal )
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask, temporal=False):
        # Pass the input through each encoder layer in sequence
        for layer in self.layers:
            nan_count = torch.sum(torch.isnan(x)).item()
            #print( 'x nan values', nan_count ) 
            x = layer(x, key_padding_mask, temporal)
            # nan_count = torch.sum(torch.isnan(x)).item()
            #print( 'x nan values', nan_count ) 
        return x
    
class EdgeUpdateNetwork(nn.Module):
    def __init__(self,  num_features, dropout=0.0  ):
        super(EdgeUpdateNetwork, self).__init__()
        
        self.num_features = num_features
        self.dropout = dropout
        
        

        self.cnn = nn.Conv2d(
                in_channels= num_features,
                out_channels= 16,
                kernel_size=1,
                bias=False)

        self.batch = nn.BatchNorm2d(num_features= 32   )
        self.relu = nn.LeakyReLU()
        self.dr = nn.Dropout2d(p=.2)

        self.cnn_last = nn.Conv2d(
                in_channels= 16,
                out_channels= 1,
                kernel_size=1,
                bias=False)
        

        

    def forward(self, node_source, node_target, temporal=False, padSource = None, padTarget = None):
        """
        node_source: Tensor of shape (batch_size, N_s, F)
        node_target: Tensor of shape (batch_size, N_t, F)
        Returns:
            source_to_target_edge: Tensor of shape (batch_size, N_s, N_t)
        """
        N_s, S, F = node_source.size()
        N_t = node_target.size(0)

        
     
        x_i = node_source.permute( 1, 0, 2).unsqueeze(2)
        #print( 'dim matrix', x_i.shape )
        x_j = node_target.permute( 1, 0, 2).unsqueeze(1)
        #print( 'dim matrix', x_j.shape )

        


        # difference in 4D -> output is S, Ns, Nt, F
        x_ij =  torch.abs(x_i - x_j ) # 
        #print( 'dim matrix', x_ij.shape )
        #print( x_ij.shape ,x_ij.permute( 2,0,1 ).unsqueeze(0).shape )


        # # getting zero mask for temporal ones
        # if temporal:
           
        #     # element wise multliplication between the two wildcares
        #     maskZeros = padSource.unsqueeze(dim=1) *  padTarget.unsqueeze(dim=0)
            
        #     maskZeros = (~maskZeros).float().repeat(1, 1, F) # to have same dimnesions
        #     # print( maskZeros.shape, x_ij.shape )
        #     #cleaning distances
        #     x_ij = x_ij * maskZeros
          
            

       

        # expanding again features
       
        #out = self.fc2(x_ij).squeeze() #outout: N_s, N_t,1
        out = self.cnn( x_ij.permute( 0, 3,1,2 )  )
        
        out = self.batch( out  )
        #print( out.shape )
        out = self.relu( out  )
        #print( out.shape )
        out = self.dr(out  )
        #print( out.shape )
        out = self.cnn_last(out  ) # S, 1, Ns, Nt

        out = out.permute( 0, 2,3,1 ).view( S,N_s, N_t)

        #print( 'dim matrix', out.shape )
        
        
        #print( out.shape )

        sim_val = torch.sigmoid(out)

        # Reshape back to edge matrix
      
        # sim_val = out.permute(2, 0, 1)  # Shape: (S, N_s, N_t)

        #print( sim_val.shape )
        source_to_target_edge = torch.nn.functional.normalize(sim_val, p=1, dim=1) # first for source - rows i
        source_to_target_edge = torch.nn.functional.normalize(source_to_target_edge, p=1, dim=2) # 2nd for target - columns j

        #print( 'source_to_target_edge', source_to_target_edge.shape )

        return source_to_target_edge
    
class NodeUpdateNetwork(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # Define layers
        # self.fc = nn.Linear(in_features * 2, out_features)
        # self.batchN = nn.BatchNorm1d(out_features)
        # self.relu = nn.LeakyReLU() # nn.ReLU()

        # # Optional dropout
        # if self.dropout > 0:
        #     self.dropout_layer = nn.Dropout(p=self.dropout)
        # else:
        #     self.dropout_layer = None

       


        self.cnn  = nn.Sequential(
                nn.Conv2d(
                in_channels= 2*in_features,
                out_channels= 16,
                kernel_size=1,
                bias=False), 

                nn.BatchNorm2d(num_features= 16   ),
                nn.LeakyReLU(),
                nn.Dropout2d(p=.2),

               nn.Conv2d(
                        in_channels= 16,
                        out_channels= out_features,
                        kernel_size=1,
                        bias=False) )

    def forward(self, node_source, node_target, source_to_target_edge, temporal=False, padSource = None, padTarget = None):
        """
        node_source: Tensor of shape ( N_s, F)
        node_target: Tensor of shape ( N_t, F)
        source_to_target_edge: Tensor of shape ( N_s, N_t)
        Returns:
            updated_node_source: Tensor of shape ( N_s, out_features)
            updated_node_target: Tensor of shape ( N_t, out_features)
        """
        
    
        N_s, S, F = node_source.size()
        N_t = node_target.size(0)

        
        node_source = node_source.permute( 1, 0, 2) # S, N_s, F
        #print( 'dim matrix', x_i.shape )
        node_target = node_target.permute( 1, 0, 2) # S, N_t, F


        # matrix is S, N_s, N_t

        # print( source_to_target_edge.shape, node_target.shape) 
        # Aggregate messages from target to source
        source_agg = torch.bmm(source_to_target_edge, node_target)  # Shape: (N_s, F)
        
        # print('aaa', source_input.shape  )
        
        # Aggregate messages from source to target
        target_agg = torch.bmm(source_to_target_edge.permute(0, 2, 1), node_source)  # Shape: ( N_t, F)

        #print( source_agg.shape, target_agg.shape ) 

        # # Concatenate with source node features
        source_input = torch.cat([node_source, source_agg], dim=2)  # Shape: (S,  N_s, 2F)
        
        # # Concatenate with target node features
        target_input = torch.cat([node_target, target_agg], dim=2)  # Shape: (S,  N_t, 2F)

        #print( source_input.shape, target_input.shape ) 

        # if temporal:

        #     maskSource = (~padSource).float().unsqueeze(-1).repeat(1, 1, 2*F) 
        #     maskTarget = (~padTarget).float().unsqueeze(-1).repeat(1, 1, 2*F)

        #     source_input = source_input*maskSource.reshape(N_s,-1)
        #     target_input = target_input*maskTarget.reshape(N_t,-1)

        #source_input = source_agg + node_source
        #target_input= target_agg + node_target


        #source_input = source_input.reshape( N_s, S,-1)
        #target_input= target_input.reshape( N_t, S,-1)

        #print( source_input.shape) 
        # Update source node features
        # out_source = self.fc(source_input)  # Shape: ( N_s, out_features)
        # # Update target node features
        # out_target = self.fc(target_input)  # Shape: (batch_size, N_t, out_features)

        cnn_source_input = source_input.permute(0,2,1).unsqueeze(-1) #S,2F,Ns,1
        cnn_target_input = target_input.permute(0,2,1).unsqueeze(-1) #S,2F,Nt,1

        #print( cnn_source_input.shape, cnn_target_input.shape )

        out_source = self.cnn(cnn_source_input).squeeze() #S,F,Ns
        out_target =  self.cnn(cnn_target_input).squeeze() #S,F,Nt

        #print( 'out cnn', out_source.shape, out_target.shape )

        out_source = out_source.permute( 2,0,1) # Ns, S,F
        out_target = out_target.permute( 2,0,1) # Nt, S,F

        #print( 'out out', out_source.shape, out_target.shape )

       
     
        #print('aaa', out_source.shape  )
        #print('aaa', out_target.shape  )

        return out_source, out_target
    


windowsToUse = 66
class model_0(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        

        self.transformer_encoder = TransformerEncoderStack( num_layers=num_layers,
                                                            embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            expansion=expansion,
                                                            drop_p=drop_p, temporal=False )

        self.edge_update_network = EdgeUpdateNetwork(num_features=5)
        self.node_update_network = NodeUpdateNetwork(in_features=5, out_features=5 )

        self.transformer_encoder_temporal = TransformerEncoderStack( num_layers=num_layers,
                                                            embed_dim=310,
                                                            num_heads=2,
                                                            expansion=expansion,
                                                            drop_p=drop_p, temporal=True  )

        self.edge_update_network_temporal = EdgeUpdateNetwork(num_features=310)
        self.node_update_network_temporal = NodeUpdateNetwork(in_features=310, out_features=310)

        self.norm_spatial = nn.LayerNorm( 5)
        self.norm_temporal = nn.LayerNorm( 310 )

   
    
    def forward(self, source, target, key_padding_mask_source_temporal, key_padding_mask_target_temporal  ):

        
        # print( source.shape, target.shape ) 
        source = rearrange(source, "b w f -> (b w) f", w = windowsToUse, f = 310)
        target = rearrange(target, "b w f -> (b w) f", w = windowsToUse, f = 310)

        # spatial
        source = rearrange(source, "b (c f) -> b c f", c = 62, f = 5)
        target = rearrange(target, "b (c f) -> b c f", c = 62, f = 5)

        # Avoiding zero padding effect on the spatial
        mask_zeros_source = key_padding_mask_source_temporal.reshape(-1,)
        mask_zeros_target = key_padding_mask_target_temporal.reshape(-1,)

        # transformer encoding 
        source_tr = self.transformer_encoder(source[~mask_zeros_source], key_padding_mask_source_temporal)
        target_tr = self.transformer_encoder(target[~mask_zeros_target], key_padding_mask_target_temporal)

               
        # # Reshape for BP
        # source = rearrange(source, "b c f -> b (c f)", c = 62, f = 5)
        # target = rearrange(target, "b c f -> b (c f)", c = 62, f = 5)

        #print( 'after spatial transformer', source_tr.shape, target_tr.shape  )
        source_to_target_edge = self.edge_update_network(source_tr, target_tr)
        #print( 'after spatial edge', source_to_target_edge.shape )
        out_source, out_target = self.node_update_network(source_tr, target_tr, source_to_target_edge)
        #print( 'after spatial node', out_source.shape, out_target.shape )

        # out_source = self.norm_spatial(out_source)
        # out_target = self.norm_spatial(out_target)

        out_source_all = torch.zeros_like(source)  # Create a tensor with the same shape as source
        out_source_all[~mask_zeros_source] = out_source

        out_target_all = torch.zeros_like(target)  # Create a tensor with the same shape as target
        out_target_all[~mask_zeros_target] = out_target

        out_source = out_source_all
        out_target = out_target_all

        
        ####################  temporal ################################
        out_source = rearrange(out_source, "b c f -> b (c f)", c = 62, f = 5)
        out_target = rearrange(out_target, "b c f -> b (c f)", c = 62, f = 5)
        

        out_source = rearrange(out_source, "(b w) f -> b w f", w = windowsToUse, f = 310)
        out_target = rearrange(out_target, "(b w) f -> b w f", w = windowsToUse, f = 310)

        
        # here the samples are N x win x 310
        #print( out_source.shape, out_target.shape )
        out_source = self.transformer_encoder_temporal(out_source , key_padding_mask_source_temporal, temporal=True ) #shape here is [8, 74, 310]
        out_target = self.transformer_encoder_temporal(out_target, key_padding_mask_target_temporal , temporal=True )
        

        
        #print( 'after temporal transformer', out_source.shape )
        

        
        # # Doing the BP
        intersection = (~key_padding_mask_source_temporal) * (~key_padding_mask_target_temporal) # B, W

        maxWindows = torch.sum( intersection, dim=1 ).min(   ).to('cpu').numpy()

        
        source_to_target_edge_agg = self.edge_update_network_temporal(out_source[:,:maxWindows], out_target[:,:maxWindows], temporal=True, padSource = key_padding_mask_source_temporal, padTarget = key_padding_mask_target_temporal)
        #print( 'after temporal edge', torch.mean( source_to_target_edge_agg, dim=0).shape )

        meanMatrix = torch.mean( source_to_target_edge_agg, dim=0).unsqueeze(0).repeat((windowsToUse-maxWindows), 1, 1)

        source_to_target_edge_agg = torch.concat([source_to_target_edge_agg, meanMatrix ])

        #print( source_to_target_edge_agg.shape ) 

        # out_source_all = torch.zeros_like(out_source)  # Create a tensor with the same shape as source
        # out_source_all[maxWindows] = out_source_bp

        # out_target_all = torch.zeros_like(out_target)  # Create a tensor with the same shape as target
        # out_target_all[:,:maxWindows]= out_target_bp

        out_source, out_target = self.node_update_network_temporal(out_source, out_target, source_to_target_edge_agg, temporal=True, padSource = key_padding_mask_source_temporal, padTarget = key_padding_mask_target_temporal)

        
        # out_source_all = torch.zeros_like(out_source)  # Create a tensor with the same shape as source
        # out_source_all[:,:maxWindows] = out_source_bp

        # out_target_all = torch.zeros_like(out_target)  # Create a tensor with the same shape as target
        # out_target_all[:,:maxWindows]= out_target_bp


        # # residual connection
        # out_source = out_source + out_source_all
        # out_target = out_target + out_target_all

        # #print(  'total windows', out_source.shape )
        

        mask_weights_s = (~key_padding_mask_source_temporal).unsqueeze(-1).float()
        #print( mask_weights_s )
        sum_output = torch.sum( out_source, dim=1)
        valid_counts = torch.sum( mask_weights_s, dim=1)
        out_source= sum_output / (valid_counts + 1e-8)

        mask_weights_t = (~key_padding_mask_target_temporal).unsqueeze(-1).float()
        sum_output = torch.sum( out_target, dim=1)
        valid_counts = torch.sum( mask_weights_t, dim=1)
        out_target= sum_output / (valid_counts + 1e-8)





        return out_source, out_target




class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # Store alpha for use in backward pass
        return x.view_as(x)  # Identity function

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by alpha
        return grad_output.neg() * ctx.alpha, None




class DomainAdaptationModel(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs ):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = model_0(num_layers=num_layers,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        expansion=expansion,
                        drop_p=drop_p,
                        )

        self.class_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 3)
                                             )


        self.domain_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62 ),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 2)
                                             )

    def forward(self, x_source, x_target, key_padding_mask_source, key_padding_mask_target, alpha):
        # Feature extraction
        #print('before embedding', x.shape)
        source_embeddings, target_embeddings = self.feature_extractor(x_source, x_target, 
                                            key_padding_mask_source, key_padding_mask_target )
        #print('after embedding', embeddings.shape)
        
      
        # gradient reversal layer (backward gradients will be reversed)
        reverse_feature_source = GradientReversalFunction.apply(source_embeddings, alpha)
        domain_output_source = self.domain_classifier(reverse_feature_source)
 
        reverse_feature_target = GradientReversalFunction.apply(target_embeddings, alpha)
        domain_output_target = self.domain_classifier(reverse_feature_target)


        # pass features to labels classifier
        class_output_source   = self.class_classifier(source_embeddings  )
        class_output_target  = self.class_classifier(target_embeddings  )

        
        return class_output_source, class_output_target, domain_output_source, domain_output_target, source_embeddings, target_embeddings


    


###################################################################################################################
############### SPATIAL - removing temporal ########################################################


class model_0_spatial(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        

        self.transformer_encoder = TransformerEncoderStack( num_layers=num_layers,
                                                            embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            expansion=expansion,
                                                            drop_p=drop_p, temporal=False )

        self.edge_update_network = EdgeUpdateNetwork(num_features=5)
        self.node_update_network = NodeUpdateNetwork(in_features=5, out_features=5 )

      

     
   
    
    def forward(self, source, target, key_padding_mask_source_temporal, key_padding_mask_target_temporal  ):

        
        # print( source.shape, target.shape ) 
        source = rearrange(source, "b w f -> (b w) f", w = windowsToUse, f = 310)
        target = rearrange(target, "b w f -> (b w) f", w = windowsToUse, f = 310)

        # spatial
        source = rearrange(source, "b (c f) -> b c f", c = 62, f = 5)
        target = rearrange(target, "b (c f) -> b c f", c = 62, f = 5)

        # Avoiding zero padding effect on the spatial
        mask_zeros_source = key_padding_mask_source_temporal.reshape(-1,)
        mask_zeros_target = key_padding_mask_target_temporal.reshape(-1,)

        # transformer encoding 
        source_tr = self.transformer_encoder(source[~mask_zeros_source], key_padding_mask_source_temporal)
        target_tr = self.transformer_encoder(target[~mask_zeros_target], key_padding_mask_target_temporal)

               
        # # Reshape for BP
        # source = rearrange(source, "b c f -> b (c f)", c = 62, f = 5)
        # target = rearrange(target, "b c f -> b (c f)", c = 62, f = 5)

        #print( 'after spatial transformer', source_tr.shape, target_tr.shape  )
        source_to_target_edge = self.edge_update_network(source_tr, target_tr)
        #print( 'after spatial edge', source_to_target_edge.shape )
        out_source, out_target = self.node_update_network(source_tr, target_tr, source_to_target_edge)
        #print( 'after spatial node', out_source.shape, out_target.shape )

        # out_source = self.norm_spatial(out_source)
        # out_target = self.norm_spatial(out_target)

        out_source_all = torch.zeros_like(source)  # Create a tensor with the same shape as source
        out_source_all[~mask_zeros_source] = out_source

        out_target_all = torch.zeros_like(target)  # Create a tensor with the same shape as target
        out_target_all[~mask_zeros_target] = out_target

        out_source = out_source_all
        out_target = out_target_all

        
        ####################  temporal ################################
        out_source = rearrange(out_source, "b c f -> b (c f)", c = 62, f = 5)
        out_target = rearrange(out_target, "b c f -> b (c f)", c = 62, f = 5)
        

        out_source = rearrange(out_source, "(b w) f -> b w f", w = windowsToUse, f = 310)
        out_target = rearrange(out_target, "(b w) f -> b w f", w = windowsToUse, f = 310)

        
       

        mask_weights_s = (~key_padding_mask_source_temporal).unsqueeze(-1).float()
        #print( mask_weights_s )
        sum_output = torch.sum( out_source, dim=1)
        valid_counts = torch.sum( mask_weights_s, dim=1)
        out_source= sum_output / (valid_counts + 1e-8)

        mask_weights_t = (~key_padding_mask_target_temporal).unsqueeze(-1).float()
        sum_output = torch.sum( out_target, dim=1)
        valid_counts = torch.sum( mask_weights_t, dim=1)
        out_target= sum_output / (valid_counts + 1e-8)





        return out_source, out_target








class DomainAdaptationModel_spatial(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs ):
        super(DomainAdaptationModel_spatial, self).__init__()
        self.feature_extractor = model_0_spatial(num_layers=num_layers,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        expansion=expansion,
                        drop_p=drop_p,
                        )

        self.class_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 3)
                                             )


        self.domain_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62 ),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 2)
                                             )

    def forward(self, x_source, x_target, key_padding_mask_source, key_padding_mask_target, alpha):
        # Feature extraction
        #print('before embedding', x.shape)
        source_embeddings, target_embeddings = self.feature_extractor(x_source, x_target, 
                                            key_padding_mask_source, key_padding_mask_target )
        #print('after embedding', embeddings.shape)
        
      
        # gradient reversal layer (backward gradients will be reversed)
        reverse_feature_source = GradientReversalFunction.apply(source_embeddings, alpha)
        domain_output_source = self.domain_classifier(reverse_feature_source)
 
        reverse_feature_target = GradientReversalFunction.apply(target_embeddings, alpha)
        domain_output_target = self.domain_classifier(reverse_feature_target)


        # pass features to labels classifier
        class_output_source   = self.class_classifier(source_embeddings  )
        class_output_target  = self.class_classifier(target_embeddings  )

        
        return class_output_source, class_output_target, domain_output_source, domain_output_target, source_embeddings, target_embeddings


    
###################################################################################################################
############### TEMPORAL  - removing spaial ########################################################


class model_temporal(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        

    

        self.transformer_encoder_temporal = TransformerEncoderStack( num_layers=num_layers,
                                                            embed_dim=310,
                                                            num_heads=num_heads,
                                                            expansion=expansion,
                                                            drop_p=drop_p, temporal=True  )

        self.edge_update_network_temporal = EdgeUpdateNetwork(num_features=310)
        self.node_update_network_temporal = NodeUpdateNetwork(in_features=310, out_features=310)

       
   
    
    def forward(self, source, target, key_padding_mask_source_temporal, key_padding_mask_target_temporal  ):

        
        

        
        ####################  temporal ################################

        
        # here the samples are N x win x 310
        #print( out_source.shape, out_target.shape )
        out_source = self.transformer_encoder_temporal(source , key_padding_mask_source_temporal, temporal=True ) #shape here is [8, 74, 310]
        out_target = self.transformer_encoder_temporal(target, key_padding_mask_target_temporal , temporal=True )
        

        
        #print( 'after temporal transformer', out_source.shape )
        

        
        # # Doing the BP
        intersection = (~key_padding_mask_source_temporal) * (~key_padding_mask_target_temporal) # B, W

        maxWindows = torch.sum( intersection, dim=1 ).min(   ).to('cpu').numpy()

        
        source_to_target_edge_agg = self.edge_update_network_temporal(out_source[:,:maxWindows], out_target[:,:maxWindows], temporal=True, padSource = key_padding_mask_source_temporal, padTarget = key_padding_mask_target_temporal)
        #print( 'after temporal edge', torch.mean( source_to_target_edge_agg, dim=0).shape )

        meanMatrix = torch.mean( source_to_target_edge_agg, dim=0).unsqueeze(0).repeat((windowsToUse-maxWindows), 1, 1)

        source_to_target_edge_agg = torch.concat([source_to_target_edge_agg, meanMatrix ])

        #print( source_to_target_edge_agg.shape ) 

        # out_source_all = torch.zeros_like(out_source)  # Create a tensor with the same shape as source
        # out_source_all[maxWindows] = out_source_bp

        # out_target_all = torch.zeros_like(out_target)  # Create a tensor with the same shape as target
        # out_target_all[:,:maxWindows]= out_target_bp

        out_source, out_target = self.node_update_network_temporal(out_source, out_target, source_to_target_edge_agg, temporal=True, padSource = key_padding_mask_source_temporal, padTarget = key_padding_mask_target_temporal)

        
        # out_source_all = torch.zeros_like(out_source)  # Create a tensor with the same shape as source
        # out_source_all[:,:maxWindows] = out_source_bp

        # out_target_all = torch.zeros_like(out_target)  # Create a tensor with the same shape as target
        # out_target_all[:,:maxWindows]= out_target_bp


        # # residual connection
        # out_source = out_source + out_source_all
        # out_target = out_target + out_target_all

        # #print(  'total windows', out_source.shape )
        

        mask_weights_s = (~key_padding_mask_source_temporal).unsqueeze(-1).float()
        #print( mask_weights_s )
        sum_output = torch.sum( out_source, dim=1)
        valid_counts = torch.sum( mask_weights_s, dim=1)
        out_source= sum_output / (valid_counts + 1e-8)

        mask_weights_t = (~key_padding_mask_target_temporal).unsqueeze(-1).float()
        sum_output = torch.sum( out_target, dim=1)
        valid_counts = torch.sum( mask_weights_t, dim=1)
        out_target= sum_output / (valid_counts + 1e-8)





        return out_source, out_target




class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # Store alpha for use in backward pass
        return x.view_as(x)  # Identity function

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by alpha
        return grad_output.neg() * ctx.alpha, None




class DomainAdaptationModel_temporal(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, expansion=2, drop_p=0.1, *args, **kwargs ):
        super(DomainAdaptationModel_temporal, self).__init__()
        self.feature_extractor = model_temporal(num_layers=num_layers,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        expansion=expansion,
                        drop_p=drop_p,
                        )

        self.class_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 3)
                                             )


        self.domain_classifier = nn.Sequential(#nn.Linear(in_features = 74*310, out_features = 74*62 ),
                                              #nn.GELU(),
                                              nn.Dropout(0.3),
                                              nn.Linear(in_features =310, out_features = 2)
                                             )

    def forward(self, x_source, x_target, key_padding_mask_source, key_padding_mask_target, alpha):
        # Feature extraction
        #print('before embedding', x.shape)
        source_embeddings, target_embeddings = self.feature_extractor(x_source, x_target, 
                                            key_padding_mask_source, key_padding_mask_target )
        #print('after embedding', embeddings.shape)
        
      
        # gradient reversal layer (backward gradients will be reversed)
        reverse_feature_source = GradientReversalFunction.apply(source_embeddings, alpha)
        domain_output_source = self.domain_classifier(reverse_feature_source)
 
        reverse_feature_target = GradientReversalFunction.apply(target_embeddings, alpha)
        domain_output_target = self.domain_classifier(reverse_feature_target)


        # pass features to labels classifier
        class_output_source   = self.class_classifier(source_embeddings  )
        class_output_target  = self.class_classifier(target_embeddings  )

        
        return class_output_source, class_output_target, domain_output_source, domain_output_target, source_embeddings, target_embeddings