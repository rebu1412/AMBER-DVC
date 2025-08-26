import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM
    from mamba_ssm.modules.mamba_new import Mamba as DBM
except:
    print('Please install mamba_ssm before using Mamba encoder')

class Processor(nn.Module):
    def __init__(self, 
                 hidden_dim=512, 
                 query_dim=512, 
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank='auto',
                 num_feature_levels=4,
                 frame_embedding_num=200):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.frame_embedding_num = frame_embedding_num
        self.temporal_num = torch.tensor([int(self.frame_embedding_num // (2 ** i)) for i in range(self.num_feature_levels)])
        
        # ffn for fusion
        self.ffn = nn.ModuleList([
            nn.Sequential(
                MLP(hidden_dim, hidden_dim*2, hidden_dim, 3),
            ) for _ in range(num_feature_levels-1)
        ])

        # Conv1d for query
        self.settings = [(2, 2), (4, 4), (8, 8), (16, 16)]
        self.query_convs = nn.ModuleList([ 
            nn.Sequential(
                    nn.Conv1d(hidden_dim, query_dim, kernel_size=k, stride=s, padding=0), # (k-1)//2
                    nn.GroupNorm(min(32, query_dim//4), query_dim),
                    nn.ReLU()
            ) for k, s in self.settings
        ])

        # SSM Refiners: Mamba-based refinement for each level
        # self.query_refiners = nn.ModuleList([
        #     nn.Sequential(
        #         DBM(
        #             d_model=query_dim,
        #             d_state=d_state,
        #             d_conv=d_conv,# if level_idx < 2 else 5,
        #             expand=expand,
        #             dt_rank=dt_rank
        #         ),
        #         nn.LayerNorm(query_dim)
        #     ) for _ in range(len(self.query_convs))
        # ])

    @staticmethod
    def prepare_query_inputs(res_feature_layer):
        return [res_feature.permute(0, 2, 1)for res_feature in res_feature_layer]

    def get_query_frame_indices(self, input_length, kernel_size, stride):
        output_length = (input_length - kernel_size) // stride + 1 # [(200 - 2) // 2] + 1 = 100
        start_indices = torch.arange(0, output_length) * stride
        center_indices = torch.arange(0, output_length) * stride + (kernel_size // 2) # [1, 3, 5, 7,..., 199]
        end_indices = start_indices + kernel_size
        return start_indices, center_indices, end_indices

    def compute_temporal_resolutions(self, durations):
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.float32)
        if durations.dim() == 0:
            durations = durations.unsqueeze(0)
        temporal_num = self.temporal_num.to(durations.device)
        frame_s = durations[:, None] / temporal_num[None, :]
        return frame_s
    
    def forward(self, decoded_features, durations):
        '''
        decoded_features: 
            list of [bs, T, d], e.g., [[bs, 200, d], [bs, 100, d], [bs, 50, d], [bs, 25, d]]
        '''
        # FPN
        features = [F.normalize(f, dim=-1) for f in decoded_features] # Normalize features
        fused_feature = features[-1]  # [bs, 25, d]
        for i in range(self.num_feature_levels-1, -1, -1):  # reverse loop: i = 3, 2, 1, 0 | (F_res4 -> F_res3 -> F_res2 -> F_res1)
            if i == 0:  # Highest resolution, no fusion needed
                break
            target_T = self.temporal_num[i-1].item()  # Target temporal dimension (e.g., 50, 100, 200)
            _, T, _ = fused_feature.shape
            fused_feature = fused_feature.repeat_interleave(target_T // T, dim=1)  # Upsample: [bs, T*repeat_factor, d]
            fused_feature = self.ffn[i-1](fused_feature)  # FFN: [bs, T*repeat_factor, d]
            fused_feature = fused_feature + features[i-1] # Fuse with the next resolution feature
        F_fused = fused_feature # [bs, 200, d]

        
        # Compute temporal resolutions
        frame_s = self.compute_temporal_resolutions(durations) # [bs]
        
        queries = []
        query_times = []
        query_cl = []
        # Conv1D to generate anchor query
        # F_fused = F_fused.transpose(1, 2)  # [bs, d, 200]
        # for conv in self.query_convs:
            # # Anchor Query (feature)
            # query = conv(F_fused)  # [bs, query_dim, T_out]
            # query = query.transpose(1, 2)  # [bs, T_out, query_dim]
            # queries.append(query)
        for cluster_size, stride in self.settings:
            bs, T, d = F_fused.shape
            output_length = (T - cluster_size) // stride + 1
            unfolded = F_fused.unfold(dimension=1, size=cluster_size, step=stride).permute(0, 1, 3, 2)  # [bs, output_length, cluster_size, d]
            avg_pooled = torch.mean(unfolded, dim=2)  # [bs, output_length, d]
            max_pooled = torch.max(unfolded, dim=2)[0]  # [bs, output_length, d]
            query = torch.cat([avg_pooled, max_pooled], dim=2)  # [bs, output_length, 2*d]
            queries.append(query)

            # Compute frame indices
            # k, s = conv[0].kernel_size[0], conv[0].stride[0]
            start_indices, center_indices, end_indices = self.get_query_frame_indices(
                input_length=self.frame_embedding_num, kernel_size=cluster_size, stride=stride
            )
            # Convert to seconds: [bs, T_out]
            level_times_start = frame_s[:, 0].unsqueeze(-1) * start_indices[None, :].to(frame_s.device)
            level_times_end = frame_s[:, 0].unsqueeze(-1) * end_indices[None, :].to(frame_s.device)
            level_bbox_xy = torch.stack([level_times_start, level_times_end], dim=-1)  # [bs, T_out, 2]

            # Normalize (center, length): [bs, T_out, 2]
            durations_expanded = durations[:, None].to(frame_s.device)  # [bs, 1]
            center = (level_times_start + level_times_end) / (2 * durations_expanded)  # [bs, T_out]s
            length = (level_times_end - level_times_start) / durations_expanded  # [bs, T_out]
            level_bbox_cl = torch.stack([center, length], dim=-1).float()  # [bs, T_out, 2]
            query_times.append(level_bbox_xy)
            query_cl.append(level_bbox_cl)

        # refined_queries = []
        # # Anchor feature refinement (layer-level)
        # for level, (queries, refiner) in enumerate(zip(anchor_queries, self.query_refiners)):
        #     refined = refiner(queries)  # [bs, sum(num_query_created), query_dim]
        #     refined_queries.append(refined)

        return queries, query_times, query_cl, frame_s

# class Processor(nn.Module):
#     def __init__(self, 
#                  hidden_dim=512, 
#                  query_dim=512, 
#                  d_state=16,
#                  d_conv=4,
#                  expand=2,
#                  dt_rank='auto',
#                  num_feature_levels=4,
#                  frame_embedding_num=200):
#         super().__init__()
#         # Layer processing
#         self.num_feature_levels = num_feature_levels
#         self.frame_embedding_num = frame_embedding_num
#         self.temporal_num = torch.tensor([int(self.frame_embedding_num // (2 ** i)) for i in range(self.num_feature_levels)])
#         # Anchor Query Generator: Conv1d layers for each level
#         self.query_convs = nn.ModuleList([ 
#             nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv1d(hidden_dim, query_dim, kernel_size=k, stride=s, padding=0),
#                     nn.GroupNorm(32, query_dim),
#                     nn.ReLU()
#                 ) for k, s in configs
#             ]) for configs in [     #                                                      After refined by SSM
#                 [(2, 2), (4, 4)],   # Level 1: (1.57 s/frame) 1–5s    --> (3s  -  6s)    ~ (3s  - 6s  - 9s  - 12s - 18s)
#                 [(3, 3), (5, 5)],   # Level 2: (3.15 s/frame) 5–20s   --> (6s  - 12s)    ~ (6s  - 12s - 18s - 24s - 36s)
#                 [(4, 4), (6, 6)],   # Level 3: (6.30 s/frame) 20–50s  --> (12s - 24s)    ~ (12s - 24s - 36s - 48s - 72s)
#                 [(5, 5), (7, 7)]    # Level 4: (12.6 s/frame) 50–120s --> (24s - 48s)    ~ (24s - 48s - 72s - 96s - 144s)
#             ]
#         ])
#         # SSM Refiners: Mamba-based refinement for each level
#         self.query_refiners = nn.ModuleList([
#             nn.Sequential(
#                 DBM(
#                     d_model=query_dim,
#                     d_state=d_state,
#                     d_conv=d_conv,# if level_idx < 2 else 5,
#                     expand=expand,
#                     dt_rank=dt_rank
#                 ),
#                 nn.LayerNorm(query_dim)
#             ) for _ in range(num_feature_levels)
#         ])
        

#     @staticmethod
#     def prepare_query_inputs(res_feature_layer):
#         return [res_feature.permute(0, 2, 1)for res_feature in res_feature_layer]

#     def get_query_frame_indices(self, input_length, kernel_size, stride):
#         """
#         Compute frame indices for the start, center, and end of each query.
#         Args:
#             input_length (int): Input temporal dimension (T_l).
#             kernel_size (int): Kernel size of the convolution.
#             stride (int): Stride of the convolution.
#         Returns:
#             tuple: (start_indices, center_indices, end_indices) for each query.
#         """
#         output_length = (input_length - kernel_size) // stride + 1 # [(200 - 2) // 2] + 1 = 100
#         start_indices = torch.arange(0, output_length) * stride
#         center_indices = torch.arange(0, output_length) * stride + (kernel_size // 2) # [1, 3, 5, 7,..., 199]
#         end_indices = start_indices + kernel_size
#         return start_indices, center_indices, end_indices

#     def compute_temporal_resolutions(self, durations):
#         """
#         Compute temporal resolutions (seconds per frame) for each video at each level.
#         frame --> second
#         """
#         if not isinstance(durations, torch.Tensor):
#             durations = torch.tensor(durations, dtype=torch.float32)
#         if durations.dim() == 0:
#             durations = durations.unsqueeze(0)
#         temporal_num = self.temporal_num.to(durations.device)
#         frame_s = durations[:, None] / temporal_num[None, :]
#         return frame_s
    
#     def forward(self, decoded_features):
#         print(self.ffn)

#     def forward(self, decoded_features, durations):
#         """
#         Args:
#             decoded_features: List of feature maps from BaseEncoder [bs, hidden_dim, T_l]
#                              (T_l = 200, 100, 50, 25 for Levels 1-4).
#             durations: raw durations [bs]
#         Returns:
#             refined_queries: List of refined queries [bs, num_queries, query_dim] for each level.
#             query_times: List of tensors [bs, num_queries, 2] containing start and end times in seconds.
#             query_cl: List of tensors [bs, num_queries, 2] containing normalized center (c) and length (l).
#         """
                ##############################################################################
#         # anchor_queries = []
#         # for level, (feats, convs) in enumerate(zip(decoded_features, self.query_convs)):
#         #     level_queries = [conv(feats) for conv in convs]  # List of [B, query_dim, N_lk]
#         #     level_queries = torch.cat(level_queries, dim=-1)  # [B, query_dim, sum(N_lk)]
#         #     anchor_queries.append(level_queries)
#         #     # print(f'Total anchor queries for Level {level + 1}: {level_queries.size(-1)}')
#         # anchor_queries = self.prepare_query_inputs(anchor_queries) # [bs, num_queries, d]
        
#         # refined_queries = []
#         # for level, (queries, refiner) in enumerate(zip(anchor_queries, self.query_refiners)):
#         #     refined = refiner(queries)        # [B, sum(N_lk), query_dim]
#         #     refined_queries.append(refined)
#         # return refined_queries
                ##############################################################################
#         frame_s = self.compute_temporal_resolutions(durations) # [bs, num_feature_levels]
#         anchor_queries = []
#         query_times = []
#         query_cl = []

#         for level, (feats, convs) in enumerate(zip(decoded_features, self.query_convs)): # per resolution
#             level_queries = []
#             level_frame_start = []
#             level_frame_end = []
#             input_length = feats.size(-1)  # F_level (200, 100, 50, 25)
#             # batch_size = feats.size(0)

#             for conv_idx, conv in enumerate(convs): # per conv1d
#                 # Anchor Query (feature)
#                 queries = conv(feats)

#                 # Compute frame indices for start, center, and end
#                 start_frame_idx, _, end_frame_idx = self.get_query_frame_indices(input_length, conv[0].kernel_size[0], conv[0].stride[0])

#                 level_queries.append(queries) # [bs, query_dim, num_query_created]
#                 level_frame_start.append(start_frame_idx) # [num_query_created]
#                 level_frame_end.append(end_frame_idx) # [num_query_created]
            
#             # bbox Frame
#             level_queries = torch.cat(level_queries, dim=-1)  # [bs, query_dim, sum(num_query_created)]
#             level_frame_start = torch.cat(level_frame_start, dim=0)  # [sum(num_query_created)]
#             level_frame_end = torch.cat(level_frame_end, dim=0)  # [sum(num_query_created)]

#             # Convert to seconds: [bs, sum(num_query_created)]
#             level_times_start = frame_s[:, level].unsqueeze(-1) * level_frame_start[None, :].to(frame_s.device) # [bs, sum(num_query_created)]
#             level_times_end = frame_s[:, level].unsqueeze(-1) * level_frame_end[None, :].to(frame_s.device) # [bs, sum(num_query_created)]
#             level_bbox_xy = torch.stack([level_times_start, level_times_end], dim=-1) # [bs, sum(num_query_created), 2]

#             # Normalize (c, l): [bs, sum(num_query_created), 2]
#             durations_expanded = durations[:, None].to(frame_s.device)  # [bs, 1]
#             center = (level_times_start + level_times_end) / (2 * durations_expanded)  # [bs, sum(N_lk)]
#             length = (level_times_end - level_times_start) / durations_expanded  # [bs, sum(N_lk)]
#             level_bbox_cl = torch.stack([center, length], dim=-1).float()  # [bs, sum(num_query_created), 2]

#             anchor_queries.append(level_queries)
#             query_times.append(level_bbox_xy)
#             query_cl.append(level_bbox_cl)

#             # print(f'Level {level + 1}: {level_queries.size(-1)} queries ')
#             # print(f'Duration: {durations[0]}s ')
#             # print(f'times: {level_bbox_xy[0, :5].tolist()}, ')
#             # print(f'c_l: {level_bbox_cl[0, :5].tolist()}')

#         # anchor_queries = self.prepare_query_inputs(anchor_queries)
#         # return anchor_queries, query_times, query_cl
        
#         refined_queries = []
#         # Anchor feature refinement (layer-level)
#         for level, (queries, refiner) in enumerate(zip(anchor_queries, self.query_refiners)):
#             refined = refiner(queries)  # [bs, sum(num_query_created), query_dim]
#             refined_queries.append(refined)

#         return refined_queries, query_times, query_cl

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def build_processor(args):
    return Processor(
        hidden_dim=args.hidden_dim, 
        query_dim=args.hidden_dim, 
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_rank=args.dt_rank,
        num_feature_levels=args.num_feature_levels,
        frame_embedding_num=args.frame_embedding_num
    )

if __name__ == '__main__': # python -m amber.processor
    batch_size = 8
    hidden_dim = 512
    query_dim = 512
    num_feature_levels = 4
    d_state = 16
    d_conv=4
    expand = 2
    dt_rank = "auto"

    # Create dummy input tensors
    decoded_features = [
        torch.randn(batch_size, 200, query_dim).cuda(),  # Level 1
        torch.randn(batch_size, 100, query_dim).cuda(),  # Level 2
        torch.randn(batch_size, 50, query_dim).cuda(),   # Level 3
        torch.randn(batch_size, 25, query_dim).cuda()    # Level 4
    ]
    # decoded_features = Processor.prepare_query_inputs(decoded_features)

    # Decoder
    model = Processor(
        hidden_dim=hidden_dim, 
        query_dim=query_dim, 
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dt_rank=dt_rank,
        num_feature_levels=num_feature_levels
        ).cuda()

    anchor_queries = model(decoded_features)

    # Print output shapes for verification
    for level, queries in enumerate(anchor_queries):
        print(f"Level {level + 1}: {queries.shape}")
