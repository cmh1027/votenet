import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetFPModule
import pointnet2_utils

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, batch_first=False, ratio=1, kv_len=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert(d_model % nhead == 0)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.batch_first = batch_first
        self.pe = nn.Linear(3, 1)
        if ratio == 1:
            self.linear_k = nn.Identity()
            self.linear_v = nn.Identity()
        else:
            self.linear_k = nn.Conv1d(kv_len, kv_len//ratio, 1)
            self.linear_v = nn.Conv1d(kv_len, kv_len//ratio, 1)
        self.ratio = ratio
    
    def forward(self, query, key, value, query_xyz=None, key_xyz=None): # (seq_len, n_batch, d_emb)
        if self.batch_first is False: # (n_batch, seq_len, d_emb)
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            if query_xyz is not None and key_xyz is not None:
                query_xyz = query_xyz.permute(1, 0, 2)
                key_xyz = key_xyz.permute(1, 0, 2)
        n_batch, seq_len_tgt, d_emb = query.shape
        _, seq_len_mem, _ = key.shape
        key, value = self.linear_k(key), self.linear_v(value)
        query = self.W_q(query).view(n_batch, seq_len_tgt, self.nhead, -1).transpose(1, 2) # (n_batch, h, seq_len, d_k)
        key = self.W_k(key).view(n_batch, seq_len_mem // self.ratio, self.nhead, -1).transpose(1, 2)
        value = self.W_v(value).view(n_batch, seq_len_mem // self.ratio, self.nhead, -1).transpose(1, 2)
        out, attn_map = self.calculate_attention(query, key, value, query_xyz, key_xyz)
        out = self.w_o(out.transpose(1, 2).reshape(n_batch, seq_len_tgt, -1)) # (n_batch, seq_len, d_emb)
        attn_map = F.softmax(attn_map.mean(dim=-3), dim=-1) # (n_batch, h, len1, len2) -> # (n_batch, len1, len2)
        if self.batch_first is False:
            out = out.permute(1, 0, 2)
        return out, attn_map
        
    def calculate_attention(self, query, key, value, query_xyz, key_xyz):
        # query, key, value : (n_batch, h, seq_len, d_k)
        batch, d_k = key.shape[0], key.shape[-1]
        scaling = math.sqrt(d_k)
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # QK^T (n_batch, len1, len2)
        attention_score = attention_score / scaling
        if query_xyz is not None and key_xyz is not None:
            assert self.ratio == 1, "new PE is not compatible with LinFormer"
            query_xyz = query_xyz.unsqueeze(-2) # (n_batch, len1, 3)
            key_xyz = key_xyz.unsqueeze(-3) # (n_batch, len2, 3)
            attention_score = attention_score + self.pe(query_xyz - key_xyz).squeeze(-1).unsqueeze(-3) # (n_batch, len1, len2)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
        return out, attention_prob
        


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", prenorm=True, ratio=1, src_len=2048, new_pe=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, ratio=ratio, kv_len=src_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1_pre = nn.Identity()
        self.norm1_post = nn.Identity()
        self.norm2_pre = nn.Identity()
        self.norm2_post = nn.Identity()
        if prenorm is True:
            self.norm1_pre = nn.LayerNorm(d_model)
            self.norm2_pre = nn.LayerNorm(d_model)
        else:
            self.norm1_post = nn.LayerNorm(d_model)
            self.norm2_post = nn.LayerNorm(d_model)            
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.activation = nn.ReLU(inplace=True)
        self.new_pe = new_pe
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, xyz=None, src_mask = None, src_key_padding_mask = None): # src : ns B*np D / xyz : ns B*np 3
        query_xyz, key_xyz = None, None
        if self.new_pe is True:
            query_xyz, key_xyz = xyz, xyz
        src = self.norm1_pre(src)
        src2, attn_map = self.self_attn(src, src, src, query_xyz=query_xyz, key_xyz=key_xyz)
        src = src + self.dropout1(src2)
        src = self.norm1_post(src)
        src = self.norm2_pre(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2_post(src)
        return src, attn_map


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", prenorm=True, ratio=1, tgt_len=2048, mem_len=20000, new_pe=False):
        
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, ratio=ratio, kv_len=tgt_len)
        self.multihead_attn = MultiheadAttention(d_model, nhead, ratio=ratio, kv_len=mem_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1_pre = nn.Identity()
        self.norm1_post = nn.Identity()
        self.norm2_pre = nn.Identity()
        self.norm2_post = nn.Identity()
        self.norm3_pre = nn.Identity()
        self.norm3_post = nn.Identity()
        if prenorm is True:
            self.norm1_pre = nn.LayerNorm(d_model)
            self.norm2_pre = nn.LayerNorm(d_model)
            self.norm3_pre = nn.LayerNorm(d_model)
        else:
            self.norm1_post = nn.LayerNorm(d_model)
            self.norm2_post = nn.LayerNorm(d_model) 
            self.norm3_post = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)
        self.activation = nn.ReLU(inplace=True)
        self.new_pe = new_pe

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, xyz_tgt=None, xyz_mem=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        query_xyz, key_xyz = None, None
        if self.new_pe is True:
            query_xyz, key_xyz = xyz_tgt, xyz_mem
    
        tgt = self.norm1_pre(tgt)
        tgt2, attn_map = self.self_attn(tgt, tgt, tgt, query_xyz=query_xyz, key_xyz=query_xyz)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1_post(tgt)

        tgt = self.norm2_pre(tgt)
        memory = self.norm_mem(memory)
        tgt2, attn_map = self.multihead_attn(tgt, memory, memory, query_xyz=query_xyz, key_xyz=key_xyz)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2_post(tgt)
        tgt = self.norm3_pre(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3_post(tgt)
        return tgt, attn_map

class LocalTransformer(nn.Module):

    def __init__(self, npoint, radius, nsample, dim_feature, dim_out, nhead=4, num_layers=2, ratio=1, drop=0.0, prenorm=True, refinement=0, new_pe=False):
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.nc_in = dim_feature
        self.nc_out = dim_out
        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=False)
        self.ratio = ratio
        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )
        self.new_pe = new_pe
        self.chunk = nn.ModuleList()
        for _ in range(num_layers):
            self.chunk.append(TransformerEncoderLayer(  d_model=self.nc_in, 
                                                        dim_feedforward=2 * self.nc_in, 
                                                        dropout=drop, 
                                                        nhead=nhead, 
                                                        prenorm=prenorm, 
                                                        ratio=ratio,
                                                        src_len=nsample,
                                                        new_pe=new_pe))
        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)
        self.refinement = refinement
        
    def forward(self, xyz, features):
        xyz_flipped = xyz.transpose(1, 2).contiguous()

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2) # B, np, 3
        group_features, group_xyz = self.grouper(xyz.contiguous(), new_xyz.contiguous(), features.contiguous()) # (B, C, npoint, nsample) (B, 3, npoint, nsample) 
        
        B = group_xyz.shape[0]
        if self.new_pe is False: 
            input_features = group_features + self.pe(group_xyz)
        else:
            input_features = group_features
        B, D, np, ns = input_features.shape # B D np ns
        # ex. (B, D, np, ns) = (2, 64, 2048, 16)
        input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1) # ns B*np D
        input_xyz = group_xyz.permute(0, 2, 1, 3).reshape(-1, 3, ns).permute(2, 0, 1) # ns B*np 3
        transformed_feats = input_features # ns B*np D
        for enc_layer in self.chunk:
            transformed_feats, attn_map = enc_layer(transformed_feats, xyz=input_xyz) # attn_map : # B*np, ns, k (k = ns // ratio)
        if self.ratio == 1:
            ################  Coordinate Refinement ################
            """
            Coordinate refinement mode
            0 : None
            1 : Select one
            2 : Mean
            * If ratio > 1, attention matrix is projected to the original shape
            """
            if self.refinement == 0:
                refined_new_xyz = new_xyz
            elif self.refinement == 1:
                weight = attn_map.mean(dim=1) # B*np, ns
                max_attn_idx = weight.argmax(dim=-1).reshape(B, -1) # B, np
                refined_new_xyz = group_xyz.permute(0, 2, 3, 1)[torch.arange(B).view(B, 1), torch.arange(np).view(1, np), max_attn_idx] # B, np, 3
            else:
                weight = attn_map.mean(dim=1) # B*np, ns
                weight = F.softmax(weight.reshape(B, np, -1), dim=-1) # B np ns
                weighted_xyz = weight.unsqueeze(-1) * group_xyz.permute(0, 2, 3, 1) # (B, np, ns, 3)
                refined_new_xyz = weighted_xyz.sum(dim=2) # (B, np, 3)
        else:
            refined_new_xyz = new_xyz
        transformed_feats = transformed_feats.permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
        output_features = self.fc(output_features).squeeze(-1)

        return refined_new_xyz, output_features, fps_idx
        

class GlobalTransformer(nn.Module):

    def __init__(self, dim_feature, dim_out, nhead=4, num_layers=2, ratio=1, src_pts=2048, drop=0.0, prenorm=True, new_pe=False):
        
        super().__init__()

        self.nc_in = dim_feature
        self.nc_out = dim_out
        self.nhead = nhead
        
        self.pe = nn.Sequential(
            nn.Conv2d(3, self.nc_in // 2, 1),
            nn.BatchNorm2d(self.nc_in // 2),
            nn.ReLU(),
            nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
        )
        self.chunk = nn.ModuleList()
        for _ in range(num_layers):
            self.chunk.append(TransformerEncoderLayer(  d_model=self.nc_in, 
                                                        dim_feedforward=2 * self.nc_in, 
                                                        dropout=drop, 
                                                        nhead=nhead, 
                                                        prenorm=prenorm, 
                                                        ratio=ratio,
                                                        src_len=src_pts,
                                                        new_pe=new_pe))
        
        self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)
        self.new_pe = new_pe

    def forward(self, xyz, features): # B np 3 / B D np
        xyz_flipped = xyz.transpose(1, 2).unsqueeze(-1)
        if self.new_pe is False:
            input_features = features.unsqueeze(-1) + self.pe(xyz_flipped)
        else:
            input_features = features.unsqueeze(-1)
        input_features = input_features.squeeze(-1).permute(2, 0, 1)
        transformed_feats = input_features
        for enc_layer in self.chunk:
            transformed_feats, attn_map = enc_layer(transformed_feats, xyz=xyz.permute(1, 0, 2))
        transformed_feats = transformed_feats.permute(1, 2, 0)
        output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)
        
        return output_features

# class LocalGlobalTransformer(nn.Module):

#     def __init__(self, dim_in, dim_out, nhead=4, num_layers=2, ratio=1, mem_pts=20000, tgt_pts=2048, drop=0.0, dim_feature=64, prenorm=True, new_pe=False):

#         super().__init__()
        
#         self.nc_in = dim_in
#         self.nc_out = dim_out
#         self.nhead = nhead
#         self.pe = nn.Sequential(
#             nn.Conv2d(3, self.nc_in // 2, 1),
#             nn.BatchNorm2d(self.nc_in // 2),
#             nn.ReLU(),
#             nn.Conv2d(self.nc_in // 2, self.nc_in, 1)
#         )
#         self.chunk = nn.ModuleList()
#         for _ in range(num_layers):
#             self.chunk.append(TransformerDecoderLayer(  d_model=self.nc_in, 
#                                                         dim_feedforward=2 * self.nc_in,
#                                                         dropout=drop, 
#                                                         nhead=nhead, 
#                                                         nc_mem=dim_feature, 
#                                                         prenorm=prenorm, 
#                                                         ratio=ratio,
#                                                         tgt_len=tgt_pts,
#                                                         mem_len=mem_pts,
#                                                         new_pe=new_pe))
#         self.fc = nn.Conv2d(self.nc_in, self.nc_out, 1)
#         self.new_pe = new_pe

#     def forward(self, xyz_tgt, xyz_mem, features_tgt, features_mem): # B np 3 / B D np
#         xyz_tgt_flipped = xyz_tgt.transpose(1, 2).unsqueeze(-1)
#         xyz_mem_flipped = xyz_mem.transpose(1, 2).unsqueeze(-1)
#         if self.new_pe is False:
#             tgt = features_tgt.unsqueeze(-1) + self.pe(xyz_tgt_flipped)
#             mem = features_mem.unsqueeze(-1) + self.pe(xyz_mem_flipped)
#         else:
#             tgt = features_tgt.unsqueeze(-1)
#             mem = features_mem.unsqueeze(-1)  
        
#         mem = mem.squeeze(-1).permute(2, 0, 1)
#         tgt = tgt.squeeze(-1).permute(2, 0, 1)
#         transformed_feats = tgt
#         for dec_layer in self.chunk:
#             transformed_feats, attn_map = dec_layer(
#                 transformed_feats, 
#                 mem, 
#                 memory_mask=None, 
#                 xyz_tgt=xyz_tgt.permute(1, 0, 2), 
#                 xyz_mem=xyz_mem.permute(1, 0, 2)
#             )
#         transformed_feats = transformed_feats.permute(1, 2, 0)
#         output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)
#         return output_features


class LocalGlobalTransformer(nn.Module):

    def __init__(self, dim_in, dim_out, nhead=4, num_layers=2, ratio=1, mem_pts=20000, tgt_pts=2048, drop=0.0, dim_feature=64, prenorm=True, new_pe=False):
        super().__init__()
        self.nc_in = dim_in
        self.nc_out = dim_out
        self.nhead = nhead
        self.pe = nn.Sequential(
            nn.Conv1d(3, self.nc_in // 2, 1),
            nn.Conv1d(self.nc_in // 2, self.nc_in, 1)
        )

        
        self.chunk = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=0.5, nhead=nhead), num_layers=num_layers
        )
        
        self.fc = nn.Conv1d(self.nc_in, self.nc_out, 1)

    def forward(self, xyz_tgt, xyz_mem, features_tgt, features_mem):

        xyz_tgt_flipped = xyz_tgt.transpose(1, 2)
        xyz_mem_flipped = xyz_mem.transpose(1, 2)
        tgt = features_tgt + self.pe(xyz_tgt_flipped)
        mem = features_mem + self.pe(xyz_mem_flipped)

        mem_mask = None
        
        mem = mem.permute(2, 0, 1)
        tgt = tgt.permute(2, 0, 1)

        transformed_feats = self.chunk(tgt, mem, memory_mask=mem_mask).permute(1, 2, 0)
        output_features = self.fc(transformed_feats)
        
        return output_features

class BasicDownBlock(nn.Module):
    def __init__(self, 
                 npoint, radius, nsample, 
                 dim_feature, dim_hid, dim_out, 
                 nhead=4, num_layers=2, 
                 ratio=1, mem_pts=20000, 
                 use_decoder=True,
                 local_drop=0.0, global_drop=0.0, decoder_drop=0.0,
                 prenorm=True,
                 refinement=0,
                 new_pe=False):
        
        super().__init__()
        use_decoder=False
        self.use_decoder = use_decoder
        self.local_chunk = LocalTransformer(npoint, radius, nsample, dim_feature, dim_hid, nhead, num_layers, ratio, local_drop, prenorm, refinement, new_pe)
        self.global_chunk = GlobalTransformer(dim_hid, dim_out, nhead, num_layers, ratio, npoint, global_drop, prenorm, new_pe)
        if use_decoder:
            self.combine_chunk = LocalGlobalTransformer(dim_hid, dim_hid, nhead, 1, ratio, mem_pts, npoint, decoder_drop, dim_feature, prenorm, new_pe)
    def forward(self, xyz, features):
        new_xyz, local_features, fps_idx = self.local_chunk(xyz, features)
        if self.use_decoder:
            combined = self.combine_chunk(new_xyz, xyz, local_features, features)
            combined += local_features
        else:
            combined = local_features
        output_feats = self.global_chunk(new_xyz, combined)

        return new_xyz.contiguous(), output_feats.contiguous(), fps_idx.contiguous()

class Pointformer(nn.Module):
    def init_weights(self, pretrained=None):
        pass

    def __init__(self, 
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(16, 16, 16, 16),
                 basic_channels=64,
                 fp_channels=((256, 256), (256, 256)),
                 num_heads=4,
                 num_layers=2,
                 ratios=(1, 1, 1, 1),
                 use_decoder=(True, True, True, True),
                 cloud_points=20000,
                 local_drop=0.0, global_drop=0.0, decoder_drop=0.0,
                 prenorm=True,
                 use_xyz=True,
                 input_feature_dim=0,
                 refinement=0,
                 new_pe=False
        ):
        super().__init__()
        bc = basic_channels
        self.num_sa = len(num_points)
        self.SA_modules = nn.ModuleList()
        self.nc_outs = []
        self.use_xyz = use_xyz
        if use_xyz is True:
            self.feature_conv = nn.Conv2d(input_feature_dim+3, bc, 1)
        else:
            self.feature_conv = nn.Conv2d(input_feature_dim, bc, 1)
        
        mem_pts = cloud_points

        for i, (np, r, ns, ratio, use) in enumerate(zip(num_points, radius, num_samples, ratios, use_decoder)):
            if i < self.num_sa - 1:
                self.SA_modules.append(BasicDownBlock(np, r, ns, bc, bc, bc * 2, num_heads, num_layers, ratio, mem_pts, use, local_drop, global_drop, decoder_drop,                                          prenorm, refinement, new_pe))
                self.nc_outs.append(bc * 2)
            else:
                self.SA_modules.append(BasicDownBlock(np, r, ns, bc, bc, bc, num_heads, num_layers, ratio, mem_pts, use, local_drop, global_drop, decoder_drop,                                              prenorm, refinement, new_pe))
                self.nc_outs.append(bc)
            
            bc = bc * 2
            mem_pts = np

        self.fp1 = PointnetFPModule(mlp=[self.nc_outs[-1] + self.nc_outs[-2], fp_channels[0][0], fp_channels[0][1]])
        self.fp2 = PointnetFPModule(mlp=[self.nc_outs[-3] + fp_channels[0][1], fp_channels[1][0], fp_channels[1][1]])

    def forward(self, points, end_points=None):
        """
            Parameters
            ----------
            points: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """ 
        xyz = points[...,0:3].contiguous()
        if self.use_xyz is True:
            features = self.feature_conv(points.transpose(1, 2).unsqueeze(-1)).squeeze(-1)
        else:
            features = self.feature_conv(points[..., 3:].transpose(1, 2).unsqueeze(-1)).squeeze(-1)
        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(batch, 1).long()
        end_points = {"sa0_xyz" : xyz, "sa0_features" : features, "sa0_inds" : indices}
        
        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](xyz, features)
            end_points[f"sa{i+1}_xyz"] = cur_xyz
            end_points[f"sa{i+1}_features"] = cur_features
            end_points[f"sa{i+1}_inds"] = torch.gather(indices, 1, cur_indices.long())
            xyz, features, indices = cur_xyz, cur_features, cur_indices

        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points

        # fp_xyz = end_points[f"sa{self.num_sa}_xyz"]
        # fp_features = end_points[f"sa{self.num_sa}_features"]
        # fp_indices = end_points[f"sa{self.num_sa}_inds"]
        # for i in range(self.num_fp):
        #     fp_features = self.FP_modules[i](
        #         end_points[f"sa{self.num_sa-i-1}_xyz"], end_points[f"sa{self.num_sa-i}_xyz"], 
        #         end_points[f"sa{self.num_sa-i-1}_features"], fp_features
        #     )
        #     end_points[f"fp{i+1}_features"] = fp_features
        #     end_points[f"fp{i+1}_xyz"] = end_points[f"sa{self.num_sa-i-1}_xyz"]
        #     end_points[f"fp{i+1}_inds"] = end_points[f"sa{self.num_sa-i-1}_inds"]

        # return end_points
