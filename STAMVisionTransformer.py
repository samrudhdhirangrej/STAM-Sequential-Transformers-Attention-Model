import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_
from torchvision.utils import save_image
from patch_embed import PatchEmbed

from time import time
from StatefulDropPath import make_DropPath_stateful, reset_StatefulDropPath
import copy

from torch.distributions.categorical import Categorical
import torch.distributed as dist


class STAMVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_size = kwargs['img_size']
        if self.img_size==224:
            self.num_patch_per_dim_glimpse = 2
        elif self.img_size==240:
            self.num_patch_per_dim_glimpse = 3
        elif self.img_size==256:
            self.num_patch_per_dim_glimpse = 4
        else:
            self.num_patch_per_dim_glimpse = None
        self.patch_size = kwargs['patch_size'] #16
        self.in_chans = 3 #kwargs['in_chans'] #3
        self.num_patch_per_dim = self.img_size//self.patch_size
        self.num_glimpse_per_dim = self.num_patch_per_dim // self.num_patch_per_dim_glimpse

        num_patches = self.patch_embed.num_patches
        self.patch_embed=PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2+num_patches, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)

    def set_mode(self, T, stepT, mlp_layers, mlp_hidden_dim):
        self.T = T
        self.stepT = stepT
        make_DropPath_stateful(self)

        # grid of glimpses is subsampled grid of patches
        pos_tokens = self.pos_embed.data[:,2:,:].clone()
        pos_tokens = pos_tokens.reshape(-1, self.num_patch_per_dim, self.num_patch_per_dim, self.embed_dim).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(self.num_glimpse_per_dim, self.num_glimpse_per_dim), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    
        self.loc_pos_embed = nn.Parameter(pos_tokens)
    
        self.location_module = nn.Sequential()
        in_dim = 3*self.embed_dim
        for l in range(mlp_layers-1):
            self.location_module.add_module('fc'+str(l), nn.Linear(in_dim, mlp_hidden_dim))
            self.location_module.add_module('bn'+str(l), nn.BatchNorm1d(mlp_hidden_dim))
            self.location_module.add_module('rl'+str(l), nn.ReLU())
            in_dim = mlp_hidden_dim
        self.location_module.add_module('fc_final', nn.Linear(in_dim, 1))
    
        self.critic = nn.Sequential()
        in_dim = 2*self.embed_dim
        for l in range(mlp_layers-1):
            self.critic.add_module('fc'+str(l), nn.Linear(in_dim, mlp_hidden_dim//4))
            self.critic.add_module('bn'+str(l), nn.BatchNorm1d(mlp_hidden_dim//4))
            self.critic.add_module('rl'+str(l), nn.ReLU())
            in_dim = mlp_hidden_dim//4
        self.critic.add_module('fc_final', nn.Linear(in_dim, 1))
        self.critic.apply(self._init_weights)
        self.critic[-1].bias.data += 0.58
    
        popart_var = self.T-1-torch.arange(self.T-1).double()
        self.register_buffer('running_mean_popart'       , torch.zeros(self.T-1))
        self.register_buffer('running_var_popart'        , popart_var)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def prepare_glimpses(self, x):

        B = x.size(0)

        x = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size) # (B, 3, 224, 224) -> (B,3,14,14,16,16)
        x = x.unfold(2,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse).unfold(3,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse)     # (B,3,14,14,16,16) -> (B,3,7,7,16,16,2,2)
        x = x.permute(0,2,3,6,7,1,4,5)        # (B,3,7,7,16,16,2,2) -> (B,7,7,2,2,3,16,16) = (B, num_glimpses_per_dim, num_glimpses_per_dim, num_patches_per_glimpse, num_patches_per_glimpse, 3, 16, 16)

        pos = self.pos_embed[:,2:,:].reshape(1,self.num_patch_per_dim,self.num_patch_per_dim,self.embed_dim)
        pos = pos.unfold(1,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse).unfold(2,self.num_patch_per_dim_glimpse,self.num_patch_per_dim_glimpse) # (1,14,14,C) -> (1,7,7,C,2,2)
        pos = pos.permute(0,1,2,4,5,3).repeat(B,1,1,1,1,1) # (1,7,7,C,2,2) -> (1,7,7,2,2,C) -> (B,7,7,2,2,C)

        return x.flatten(1,2), pos.flatten(1,2)

    def extract_features_of_glimpses(self, x_pos):
        # form input to the transformer

        B = x_pos.size(0)

        x_pos = x_pos.flatten(1,2)

        cls_pos = self.cls_token.expand(B, -1, -1) + self.pos_embed[:,:1,:]
        dist_pos = self.dist_token.expand(B, -1, -1) + self.pos_embed[:,1:2,:]
 
        x = torch.cat((cls_pos, dist_pos, x_pos), dim=1)
 
        # processing in transformer
        for blk in self.blocks:
            x = blk(x)
 
        x = self.norm(x)
        return x[:,0], x[:,1]
        
    def evaluate_locations(self, feat, future_loc, loc_pos=None):
        '''
        feat = (B, self.embed_dim)
        future_loc = (B, -stepT*t) # locations of all possible future glimpses
        '''
        B = feat.size(0)
        T = future_loc.size(1)
        loc_pos = torch.gather(self.loc_pos_embed.repeat(B, 1, 1), 1, future_loc[:,:,None].repeat(1,1,self.embed_dim))
        feat_loc_pos = torch.cat([feat[:,None,:].repeat(1,T,1), loc_pos],-1)
        unnormalized_prob = self.location_module(feat_loc_pos.flatten(0,1)).reshape(feat_loc_pos.size()[:2]) # (B, T, 2*C) -> (B, T, 1) -> (B, T)

        return unnormalized_prob

    def select_loc_from_unnormalized_prob(self, unnormalized_prob, future_loc, howmany):
        if self.location_module.training:
            unnormalized_prob = unnormalized_prob / unnormalized_prob.norm(dim=-1, keepdim=True)
            prob = (unnormalized_prob * self.loc_tau).softmax(dim=1)
            logpi = F.log_softmax(unnormalized_prob * self.loc_tau, dim=1)
            samples = torch.multinomial(prob, howmany, replacement=True) # shape: (B, howmany)
            locs = torch.gather(future_loc, 1, samples)
            logpi = torch.gather(logpi, 1, samples)
            return locs, logpi
        else:
            prob = (unnormalized_prob).softmax(dim=1)
            samples = torch.topk(prob, k=howmany, dim=1)[1] #torch.multinomial(prob, self.stepT, replacement=False)
            locs = torch.gather(future_loc, 1, samples)
            return locs


    def normalize_using_running_stats(self, measure, value, time_idx):
        B = value.numel()
        value_sum = value.sum().detach()
        if torch.distributed.is_initialized():
            dist.all_reduce(value_sum)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        value_mean = value_sum / (B*world_size)
        value_2_sum = value.pow(2).sum().detach()
        if torch.distributed.is_initialized(): dist.all_reduce(value_2_sum)
        value_2_mean = value_2_sum / (B*world_size)

        if measure=='reward':
            sigma = (value_2_mean - value_mean**2)**0.5
            value = (value - value_mean)/sigma
        else:
            beta1, beta2 = 0.99, 0.99
            getattr(self,'running_mean_'+measure)[time_idx] = getattr(self,'running_mean_'+measure)[time_idx] * beta1 + (1-beta1) * value_mean
            getattr(self,'running_var_'+measure)[time_idx] = getattr(self,'running_var_'+measure)[time_idx] * beta2 + (1-beta2) * value_2_mean
            sigma = (getattr(self,'running_var_'+measure)[time_idx] - getattr(self,'running_mean_'+measure)[time_idx]**2)**0.5
            value = (value - getattr(self,'running_mean_'+measure)[time_idx])/sigma
        return value

    def RL_loss(self, logits, value_now, value_future, logpi, teacher_logits):
        B = logits.size(0)

        # compute reward and normalize
        kld = (logits*(torch.log(logits) - torch.log(teacher_logits))).sum(-1)
        soft_reward = -kld[:,None]

        soft_reward = self.normalize_using_running_stats('reward', soft_reward, self.tempT-1)

        # build target using popart normalized values
        if self.tempT < self.T-1:
            sigma = (self.running_var_popart[self.tempT] - self.running_mean_popart[self.tempT]**2)**0.5
            value_future = sigma * value_future + self.running_mean_popart[self.tempT]
        target = soft_reward + value_future
        target = self.normalize_using_running_stats('popart', target, self.tempT-1)
        advantage = target.detach() - value_now

        actor_loss = - (logpi * advantage.detach()).mean()
        critic_loss = advantage.abs().mean()

        with torch.no_grad():
            sigma = (self.running_var_popart[self.tempT-1] - self.running_mean_popart[self.tempT-1]**2)**0.5
            value_now = sigma * value_now + self.running_mean_popart[self.tempT-1]
        return actor_loss, critic_loss

    def actor_critic(self, upto_now_loc, feat, feat_dist, x_pos, prob_fusion_teacher):
        if self.tempT < self.T: 
            B = feat.size(0)
            '''actor tasks'''
            all_loc = torch.arange(self.num_glimpse_per_dim**2)[None,...].repeat(B, 1).to(feat.device)
            candidate_loc_mask = torch.ones_like(all_loc).scatter_(1, upto_now_loc, 0.).bool()
            candidate_loc = all_loc[candidate_loc_mask].reshape(B, (self.num_glimpse_per_dim**2) - self.tempT)

            unnormalized_prob = self.evaluate_locations(torch.cat([feat, feat_dist],-1), candidate_loc)
            future_loc, logpi = self.select_loc_from_unnormalized_prob(unnormalized_prob, candidate_loc, 1)

            '''critic tasks'''
            ''' current V(S) '''
            value_now = self.critic(torch.cat([feat, feat_dist],-1).detach())

            ''' future V(S') '''
            one_step_ahead_loc = torch.cat([upto_now_loc, future_loc], 1)
            self.one_step_ahead_loc = one_step_ahead_loc
            with torch.no_grad():
                one_step_ahead_x_pos = x_pos.gather(1, one_step_ahead_loc[:,:,None,None].repeat(1,1,x_pos.size(2), x_pos.size(3)))
                feat_one_step_ahead, feat_dist_one_step_ahead = self.extract_features_of_glimpses(one_step_ahead_x_pos)
                logits_future = self.head(feat_one_step_ahead)
                logits_dist_future = self.head_dist(feat_dist_one_step_ahead)
                prob_fusion_future = (torch.softmax(logits_future, dim=-1) + torch.softmax(logits_dist_future, dim=-1))/2

                if self.tempT>=(self.T-1): # (a) t=T-1 is a terminal state for loc_module. so value_future=0.
                    value_future = torch.zeros_like(value_now)
                else:
                    value_future = self.critic(torch.cat([feat_one_step_ahead,feat_dist_one_step_ahead],-1).detach())

                
            actor_loss, critic_loss = self.RL_loss(
                                                   prob_fusion_future,
                                                   value_now, value_future, logpi,
                                                   prob_fusion_teacher
                                                   )

        else:
            actor_loss = 0
            critic_loss = 0

        return actor_loss, critic_loss

    def select_one_random_future_loc(self, upto_now_loc):
        B = upto_now_loc.size(0)
        all_loc = torch.arange(self.num_glimpse_per_dim**2)[None,...].repeat(B, 1).to(upto_now_loc.device)
        candidate_loc_mask = torch.ones_like(all_loc).scatter_(1, upto_now_loc, 0.).bool()
        candidate_loc = all_loc[candidate_loc_mask].reshape(B, (self.num_glimpse_per_dim**2) - self.tempT)
        samples = torch.randint(candidate_loc.size(1), (B,1)).to(upto_now_loc.device)
        future_loc = torch.gather(candidate_loc, 1, samples)
        one_step_ahead_loc = torch.cat([upto_now_loc, future_loc], 1)
        self.one_step_ahead_loc = one_step_ahead_loc
        return 

    def iter_init(self, x, targets):
        self.tempT = 0
        self.one_step_ahead_loc = torch.multinomial(torch.ones(x.size(0),self.num_glimpse_per_dim**2), 1, replacement=False).to(x.device)
        return


    def forward(self, x, targets, teacher_gt, teacher_dist):

        if self.training:

            B = x.size(0)
            self.tempT += 1
            reset_StatefulDropPath(self, B, x.dtype, x.device)
    
            # prepare all glimpses, all loc emebeddings for transformer and location module
            x, pos = self.prepare_glimpses(x)
            x_pos = self.patch_embed(x.flatten(1,3)) + pos.flatten(1,3) # (B, 7*7*2*2, 3, 16, 16) -> (B, 7*7*2*2, C)
            x_pos = x_pos.view(B, self.num_glimpse_per_dim**2, self.num_patch_per_dim_glimpse**2, self.embed_dim)

            upto_now_loc = self.one_step_ahead_loc

            ''' check for T>0 '''
            upto_now_x_pos = x_pos.gather(1, upto_now_loc[:,:,None,None].repeat(1,1,x_pos.size(2), x_pos.size(3)))
            feat, feat_dist = self.extract_features_of_glimpses(upto_now_x_pos)

            ''' Consistency '''
            logits_dist = self.head_dist(feat_dist)
            dist_loss = self.dist_criterion(logits_dist, teacher_gt, teacher_dist)

            ''' CLS tasks '''
            logits = self.head(feat)
            if dist_loss==torch.zeros(1).mean():
                prob_fusion = (torch.softmax(logits, dim=-1) + torch.softmax(logits_dist, dim=-1))/2
                class_loss = -torch.log(prob_fusion[range(B),targets[range(B)]]).mean()
            else:
                class_loss = self.classifier_criterion(logits, targets)

            ''' Do actor-critic if tempT < T. We optimize only classifier for the last glimpse; next glimpse location is stored in the function '''
            prob_fusion_teacher = (torch.softmax(teacher_gt, dim=-1) + torch.softmax(teacher_dist, dim=-1))/2
            actor_loss, critic_loss = self.actor_critic(upto_now_loc, feat, feat_dist, x_pos, prob_fusion_teacher)

            self.return_logits = logits.detach()
    
            return class_loss, actor_loss, critic_loss, dist_loss

        else:
            B = x.size(0)

            # prepare all glimpses, all loc emebeddings for transformer and location module
            x, pos = self.prepare_glimpses(x) 
            all_logits, all_logits_dist = [], []

            all_loc = torch.arange(self.num_glimpse_per_dim**2)[None,...].repeat(B, 1).to(x.device)
            all_loc_permuted = torch.argsort(torch.rand((B, self.num_glimpse_per_dim**2), device=x.device), dim=-1)

            x_pos = self.patch_embed(x.flatten(1,3)) + pos.flatten(1,3) # (B, 7*7*2*2, 3, 16, 16) -> (B, 7*7*2*2, C)
            x_pos = x_pos.view(B, self.num_glimpse_per_dim**2, self.num_patch_per_dim_glimpse**2, self.embed_dim)

            upto_now_x_pos = []

            for t in range(0, self.T, 1):
                if t==0:
                    current_loc = all_loc_permuted[:,:1]
                    past_loc = current_loc
                else:
                    candidate_loc_mask = torch.ones_like(all_loc).scatter_(1, past_loc, 0.).bool()
                    candidate_loc = all_loc[candidate_loc_mask].reshape(B, (self.num_glimpse_per_dim**2) - t)
                    unnormalized_prob = self.evaluate_locations(torch.cat([feat, feat_dist],-1), candidate_loc)
                    current_loc = self.select_loc_from_unnormalized_prob(unnormalized_prob, candidate_loc, 1)
                    past_loc = torch.cat([past_loc, current_loc], 1)

                current_x_pos = x_pos.gather(1, current_loc[:,:,None,None].repeat(1,1,x_pos.size(2), x_pos.size(3)))
                upto_now_x_pos.append(current_x_pos)

                feat, feat_dist = self.extract_features_of_glimpses(torch.cat(upto_now_x_pos,1))
                logits = self.head(feat)
                logits_dist = self.head_dist(feat_dist)
                if ((t+1)%self.stepT)==0:
                    all_logits.append(logits)
                    all_logits_dist.append(logits_dist)

            all_logits = torch.stack(all_logits)
            all_logits_dist = torch.stack(all_logits_dist)

            class_loss = self.classifier_criterion(all_logits.flatten(0,1), targets.unsqueeze(0).repeat([all_logits.size(0)]+[1]*targets.dim()).flatten(0,1))
            if teacher_gt is not None:
                prob_fusion_teacher = (torch.softmax(teacher_gt, dim=-1) + torch.softmax(teacher_dist, dim=-1))/2
                class_loss_dist = self.dist_criterion(all_logits_dist.flatten(0,1),
                    teacher_gt.unsqueeze(0).repeat([all_logits_dist.size(0)]+[1]*prob_fusion_teacher.dim()).flatten(0,1),
                    teacher_dist.unsqueeze(0).repeat([all_logits_dist.size(0)]+[1]*prob_fusion_teacher.dim()).flatten(0,1),
                    )
            else:
                class_loss_dist = self.dist_criterion(all_logits_dist.flatten(0,1),None,None)

            return all_logits, class_loss, all_logits_dist, class_loss_dist


    def recon_glimpse(self, x, loc):

        if isinstance(loc, torch.Tensor):
            loc = [loc//self.num_glimpse_per_dim, loc%self.num_glimpse_per_dim]
     
        B = x.size(0)
        T = loc[0].size(1)

        g = torch.zeros_like(x)

        for t in range(T):
            g[range(B),loc[0][range(B),t],loc[1][range(B),t]] = x[range(B),loc[0][range(B),t],loc[1][range(B),t]]

        g = g.permute(0,1,3,2,4,5,6,7).reshape(B,self.num_patch_per_dim,self.num_patch_per_dim,self.in_chans,self.patch_size,self.patch_size) # (B,7,7,2,2,3,16,16) -> (B,7,2,7,2,3,16,16) -> (B,14,14,3,16,16)
        g = g.permute(0,1,4,2,5,3).reshape(B,self.img_size,self.img_size,self.in_chans)         # (B,14,14,3,16,16) -> (B,14,16,14,16,3) -> (B,224,224,3)
        g = g.permute(0,3,1,2)

        x = x.permute(0,1,3,2,4,5,6,7).reshape(B,self.num_patch_per_dim,self.num_patch_per_dim,self.in_chans,self.patch_size,self.patch_size) # (B,7,7,2,2,3,16,16) -> (B,7,2,7,2,3,16,16) -> (B,14,14,3,16,16)
        x = x.permute(0,1,4,2,5,3).reshape(B,self.img_size,self.img_size,self.in_chans)         # (B,14,14,3,16,16) -> (B,14,16,14,16,3) -> (B,224,224,3)
        x = x.permute(0,3,1,2)

        save_image(torch.cat([x,g],-1)[:4], 'viz.png', normalize=True, nrow=2)

        return



