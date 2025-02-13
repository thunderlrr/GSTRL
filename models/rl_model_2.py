import torch
from torch import nn
import numpy as np
from functools import partial

from afno.afno1d import AFNO1D
import random
from torch.distributions import Categorical
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .gcn import *

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.filter = AFNO1D(
            hidden_size=dim,
            num_blocks=1,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class FftNet(nn.Module):
    def __init__(
        self,
        pos_num,
        embed_dim=32,
        depth=2,
        mlp_ratio=4.0,
        representation_size=None,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        dropcls=0,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, pos_num, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if uniform_drop:
            print("using uniform droppath with expect rate", drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print("using linear droppath with expect rate", drop_path_rate * 0.5)
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    sparsity_threshold = sparsity_threshold,
                    use_fno=use_fno,
                    use_blocks=use_blocks,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

def dense_mincut_pool(x, adj, s, d, mask=None):
    r"""
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    EPS = 1e-15
    
    x = x.unsqueeze(0) if x.dim() == 2 else x
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1) # B,N,M

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

   
    s_t = s.transpose(1, 2) # B,M,N
    adj_t = adj.transpose(1,0)
    # adj: N,M
    # out1: B,M,N
    out = torch.matmul(s_t, x) # B,M,C
    out1 = [torch.sparse.mm(d, tmp) for tmp in s]
    out1 = torch.stack(out1, dim=0).permute(0, 2, 1) # B,M,N
    out_adj = torch.bmm(out1, s)
    
    # s_t = s.transpose(1, 2)
    # out = torch.matmul(s_t, x)
    # out1 = [torch.sparse.mm(adj, tmp) for tmp in s]
    # out1 = torch.stack(out1, dim=0).permute(0, 2, 1)
    # out_adj = torch.bmm(out1, s)
    
    
    # # MinCUT regularization
    mincut_num = _rank3_trace(out_adj)
    out2 = [torch.sparse.mm(d, tmp) for tmp in s]
    out2 = torch.stack(out2, dim=0).permute(0, 2, 1)
    mincut_den = _rank3_trace(torch.bmm(out2, s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)
    # return out,None,None,None
    return out, out_adj, mincut_loss, ortho_loss

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out

class GlobalNet(nn.Module):
    def __init__(self, in_channels, n_states,device,verbose=False, normalize=False):
        super(GlobalNet, self).__init__()

        self.normalize = normalize
        self.verbose = verbose
        self.n_states = n_states
        self.N = 64 # number of regions
        self.grid_adj = torch.FloatTensor(
            np.load('data/grid_adj.npy')).to(device)
        self.d = _rank3_diag(torch.einsum(
            'ijk->ij', self.grid_adj.to_dense().unsqueeze(0))).squeeze().to_sparse()

        # reduce dim
        self.conv_state = nn.Conv2d(in_channels, self.n_states, kernel_size=1)

        # projection map
        self.conv_proj = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, self.N, 1, 1, 0))

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        # ----------
        # message passing via graph convolution
        self.gcn = GCN(n_states, n_states // 2, n_states, dropout=0.2)

        self.conv_extend = nn.Conv2d(
            n_states, in_channels, kernel_size=1, bias=False)

        # should be zero initialized
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4)

    def forward(self, x):
        '''
        :param x: (b, in_channels, h, w)
        '''
        b, in_channels, h, w = x.shape
        

        # (b, in_channels, h, w) --> (b, num_state, h, w)
        #                        --> (b, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(b, self.n_states, -1)

        # (b, in_channels, h, w) --> (b, N, h, w)
        #                        --> (b, N, h*w)
        B = self.conv_proj(x).view(b, self.N, -1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (b, N, h*w) x (b, num_state, h*w)T --> (b, N, num_state)
        x_rproj_reshaped = B
        x_n_state, out_adj, mincut_loss, ortho_loss = dense_mincut_pool(x_state_reshaped.permute(0, 2, 1),
                                                                        # self.grid_adj.unsqueeze(0).repeat(b, 1, 1),
                                                                        self.grid_adj,
                                                                        B.permute(
                                                                            0, 2, 1),
                                                                        self.d)

        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (b, N, num_state) -> (b, num_state, N)
        region_out = self.gcn(x_n_state, out_adj)

        # reverse projection: interaction space -> coordinate space
        # (b, num_state, N) x (b, N, h*w) --> (b, num_state, h*w)
        x_state_reshaped = torch.matmul(region_out, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (b, num_state, h*w) --> (b, num_state, h, w)
        x_state = x_state_reshaped.view(b, self.n_states, *x.size()[2:])

        # -----------------
        # (b, num_state, h, w) -> (b, num_in, h, w)
        grid_out = self.bn(self.conv_extend(x_state))
       
        region_out = region_out.permute(0,2,1)
        grid_out = grid_out.reshape(b,h*w,-1)
        return region_out,grid_out,mincut_loss, ortho_loss



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, agent_num, init_pos, grid_size, energy, device, region_size=[4,4]):
        """
        Initialize the model
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension  
            output_size: Output dimension
            agent_num: Number of agents
            init_pos: Initial positions
            grid_size: Size of the grid world
            energy: Initial energy
            device: Computing device
            region_size: Size of each region
        """
        super(Model, self).__init__()
        
        # Network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.spatial_emb = nn.Linear(1, hidden_size)
        self.spatial_layer = FftNet(grid_size[0]*grid_size[1], hidden_size, sparsity_threshold=0.01, depth=2)
        self.global_net = GlobalNet(hidden_size*2, hidden_size*2, device)
        
        # Output layers
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.region_output = nn.Linear(hidden_size*2, output_size)
        self.grid_output = nn.Linear(hidden_size*2, output_size)
        self.score = nn.Linear(region_size[0]*region_size[1], output_size)
        
        # Model parameters
        self.grid_size = grid_size
        self.region_size = region_size
        self.agent_num = agent_num 
        self.init_pos = init_pos
        self.energy = energy
        self.device = device
        
        # Constants
        self.decrement = 0.8
        self.epsilon = 0.9
        self.region_energy_constraint = 0.6
        self.grid_energy_constraint = 0.3

    def get_energy_cost(self, agent_pos, candidate_pos):
        """Calculate energy cost between current and candidate positions"""
        agent_pos_expanded = agent_pos.unsqueeze(0).expand_as(candidate_pos)
        differences = (agent_pos_expanded - candidate_pos)**2
        squared_distances = differences.sum(dim=1)
        return torch.sqrt(squared_distances)

    def get_region_id_from_coordinate(self, x, y):
        """Convert grid coordinates to region ID"""
        regions_per_dim = self.grid_size[0] // self.region_size[0]
        return (x // self.region_size[0]) * regions_per_dim + (y // self.region_size[0])

    def forward(self, heat_maps, grid_pos, region_to_grid, centers, train=False):
        """
        Forward pass of the model
        Args:
            heat_maps: Flow density maps (B,T,H,W)
            grid_pos: Grid coordinates (H,W,2)
            region_to_grid: Mapping from regions to grid cells
            centers: Region center coordinates
            train: Training mode flag
        """
        B,T,H,W = heat_maps.shape
        
        centers = torch.tensor(centers).to(self.device)
         
        grid_pos_2d = torch.tensor(grid_pos).reshape(H*W,-1).to(self.device)
        
        grid_pos_1d = grid_pos_2d[...,0] *W + grid_pos_2d[...,1]
        
        agent_position = torch.tensor(self.init_pos).to(self.device)
        agent_positions_2d = agent_position.expand(B,self.agent_num,-1) # B,agent_num,2

        agent_positions_1d = agent_positions_2d[..., 0] * W + agent_positions_2d[..., 1]
       
        agent_energy = torch.tensor([self.energy],dtype=torch.float32).to(self.device)
        agent_energy = agent_energy.repeat(B,self.agent_num) # B,agent_num,1
     
        heat_maps_vec = torch.tensor(heat_maps,dtype=torch.float32).unsqueeze(-1).to(self.device)

        min_val = torch.min(heat_maps_vec)
        max_val = torch.max(heat_maps_vec)

        # Normalize
        heat_maps_vec = (heat_maps_vec - min_val) / (max_val - min_val)
        
        x = self.input_layer(heat_maps_vec).reshape(B*H*W,T,-1) # Shape: B*H*W,T,hidden_size
        lstm_out,(hidden,cell) = self.lstm(x)  # Shape: B*H*W,T,hidden_size
        temporal = lstm_out[:,-1 ,:].reshape(B,H,W,-1) # Shape: B,H,W,Hidden_size
        
        prediction_1d = []
        prediction_2d = []
        state_values = []
        log_probs = []

        # Iterate through time steps
        for t in range(T):
            
            t_predicted_1d = []  # Predicted positions for each frame
            t_predicted_2d = []
            t_state_values = []
            t_log_probs = []
            t_region = []
            curr_heat_maps = heat_maps_vec[:,t,:]
            curr_heat_map_vec = self.spatial_emb(curr_heat_maps.reshape(B,H*W,-1))
            curr_heat_map_vec = self.spatial_layer(curr_heat_map_vec).reshape(B,H,W,-1)
            
            combine_vec = torch.cat([curr_heat_map_vec,temporal],dim=-1).reshape(B,-1,H,W)
            region_out,grid_out,mincut_loss, ortho_loss = self.global_net(combine_vec)

            # Iterate through each batch
            for b in range(B):
                batch_predicted_2d = []
                batch_predicted = []
                batch_state_values = []
                batch_log_probs = []
                batch_region = []

                # Iterate through each agent
                for i in range(self.agent_num):
                    temp_agent_pos_2d = agent_positions_2d[b,i]
                    
                    curr_region = self.get_region_id_from_coordinate(temp_agent_pos_2d[0],temp_agent_pos_2d[1])
                    
                    region_vec = region_out[b]
                    
                    region_predicted = self.region_output(region_vec)

                    region_energy_cost = self.get_energy_cost(temp_agent_pos_2d,centers)
                    region_energy_mask = region_energy_cost > self.region_energy_constraint*agent_energy[b,i]
                    region_energy_mask[curr_region] = False
                    region_predicted[region_energy_mask] = -np.inf

                    region_log_p = torch.log_softmax(region_predicted.squeeze(-1), dim=0)
                    region_probs = region_log_p.exp()
                    
                    if(len(batch_region)>0):
                        for index in batch_region:
                            region_probs[index] = self.decrement * region_probs[index]
                    
                    if train==True:
                        if(random.random()<self.epsilon):
                            _, region_id = region_probs.max(0)
                        else:
                            multi_dist = Categorical(region_probs)
                            region_id = multi_dist.sample()
                    else:        
                        _, region_id = region_probs.max(0)
                    
                    # _,region_id = region_probs.max(0)

                    batch_region.append(region_id.item())
                    candidate_grids = torch.tensor(region_to_grid[region_id.item()]).to(self.device)
                    
                    candidate_grids_1d = candidate_grids[...,0] *W + candidate_grids[...,1]

                    grid_energy_cost = self.get_energy_cost(temp_agent_pos_2d,candidate_grids)
                    grid_energy_mask = grid_energy_cost > self.grid_energy_constraint*agent_energy[b,i]
                    # grid_vec = grid_out[b]
                    grid_vec = grid_out[b][candidate_grids_1d]
                    
                    agent_predicted = self.grid_output(grid_vec)    
                    temp_agent_pos_1d = agent_positions_1d[b,i]
                
                    agent_predicted = self.grid_output(grid_vec) # output layer
                    
                    if(len(batch_predicted)>0):
                        A = torch.tensor(batch_predicted)
                        # temp_agent_neighbor_1d
                        selected_mask = torch.full_like(agent_predicted, 0).squeeze(-1)
                        for a in A:
                            match = torch.nonzero(candidate_grids_1d == a).view(-1)
                            if match.nelement() > 0:
                                selected_mask[match[0]]=1

                        selected_mask = selected_mask == 1
                        agent_predicted[selected_mask] = -np.inf

                    agent_predicted[grid_energy_mask] = -np.inf

                    log_p = torch.log_softmax(agent_predicted.squeeze(-1), dim=0)
                    # print(log_p)
                    probs = log_p.exp()
                    # print(probs)

                    scores = self.score(probs.squeeze(-1))    
                    if train==True:
                        if(random.random()<self.epsilon):
                            _, idx = probs.max(0)
                        else:
                            multi_dist = Categorical(probs)
                            idx = multi_dist.sample()
                    else:        
                        _, idx = probs.max(0)
                    
                    # print(idx)
                    # idx = top_indices[temp_index]
                    
                    predicted_position = candidate_grids_1d[idx]
                    predicted_position_2d = candidate_grids[idx]

                    agent_energy[b,i] -= grid_energy_cost[idx].item()
                    
                    batch_predicted.append(predicted_position)
                    batch_predicted_2d.append(predicted_position_2d.squeeze(0))
                    batch_log_probs.append(log_p.squeeze(-1))
                    batch_state_values.append(scores.squeeze(-1))

                t_predicted_1d.append(torch.stack(batch_predicted,dim=0))
                t_predicted_2d.append(torch.stack(batch_predicted_2d,dim=0))
                t_log_probs.append(torch.stack(batch_log_probs,dim=0))
                t_state_values.append(torch.stack(batch_state_values,dim=0))
              
            
            agent_positions_1d = torch.stack(t_predicted_1d,dim=0).squeeze(-1)
            agent_positions_2d = torch.stack(t_predicted_2d,dim=0)
            prediction_1d.append(agent_positions_1d)
            prediction_2d.append(agent_positions_2d)       
            state_values.append(torch.stack(t_state_values,dim=0))
            log_probs.append(torch.stack(t_log_probs,dim=0))
        prediction_1d = torch.stack(prediction_1d,dim=0).reshape(B,self.agent_num,T)
        prediction_2d = torch.stack(prediction_2d,dim=0).reshape(B,self.agent_num,T,-1)
        log_probs = torch.stack(log_probs,dim=0).reshape(B,self.agent_num,T,-1)
        
        state_values = torch.stack(state_values,dim=0)
        # print(prediction_2d[0,0:2,:])
        next_state_values = state_values[1:,:].clone()
        next_state_values = torch.cat([next_state_values,torch.zeros(1,B,self.agent_num).to(self.device)])
        
        return prediction_1d, prediction_2d, state_values,next_state_values,log_probs
        # if(train==True):
        #     return prediction_1d, prediction_2d, state_values,next_state_values, log_probs,mincut_loss, ortho_loss
        # else:
        #     return prediction_1d, prediction_2d, state_values,next_state_values, log_probs
