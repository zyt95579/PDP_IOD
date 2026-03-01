import torch
import torch.nn as nn
import pdb
import copy

'''
adapted from: https://github.com/GT-RIPL/CODA-Prompt
'''


class Prompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, args=None, task_num_classes=None):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.total_classes = 80
        self._init_smart(emb_d, prompt_param)
        #print(prompt_param)
        self.task_num_classes = task_num_classes
        self.pool_sizes = [
            int(80 * (n / self.total_classes))
            for n in self.task_num_classes]
        # print(self.task_num_classes)
        if args.local_query:
            self.query_tf = nn.Sequential(
                nn.Linear(300 * 256, 300),
            )

        # ---- Shared Pool ----
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.shared_size, e_l, emb_d)
            k = tensor_prompt(self.shared_size, self.key_d)
            a = tensor_prompt(self.shared_size, self.key_d)
            p = self.gram_schmidt_shared(p)
            k = self.gram_schmidt_shared(k)
            a = self.gram_schmidt_shared(a)
            setattr(self, f'shared_p_{e}', p)
            setattr(self, f'shared_k_{e}', k)
            setattr(self, f'shared_a_{e}', a)

        # ---- Private Pool for all tasks ----
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.private_size, e_l, emb_d)
            k = tensor_prompt(self.private_size, self.key_d)
            a = tensor_prompt(self.private_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'private_p_{e}', p)
            setattr(self, f'private_k_{e}', k)
            setattr(self, f'private_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):
        # prompt basic param
        self.shared_size = int(prompt_param[0])
        self.private_size = int(80)
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4, 5]

        # ortho penalty
        self.ortho_mu = prompt_param[2]

        self.diversity_lambda = 0.15
        self.diversity_threshold = 0.2
        self.use_ddl_loss = False
        self.ddl_angle_threshold = 1.570796

    def ddl_loss(self, pa, pb, threshold):

        npa = pa.view(pa.size(0), -1)
        npb = pb.view(pb.size(0), -1)

        cosine_sim = nn.functional.cosine_similarity(npa[:, None], npb, dim=2)
        cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine_sim)
        threshold_tensor = torch.full_like(theta, threshold)

        loss = torch.sum(nn.functional.relu(threshold_tensor - theta)) * 2 / (npa.size(0) * npb.size(0))
        return loss

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.private_size / (self.n_tasks))
        s = sum(self.pool_sizes[:self.task_count]) if self.task_count > 0 else 0
        f = s + self.pool_sizes[self.task_count]

        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()

        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk

        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def gram_schmidt_shared(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        for k in range(nk):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk

        for k in range(nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def set_task_id(self, task_id=0):
        self.task_count = task_id
        print('Setting task id : ', task_id)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        if len(x_querry.shape) != 2:
            query_wt = self.query_tf(x_querry.view(x_querry.shape[0], -1))
            x_querry = x_querry * query_wt.unsqueeze(-1)
            x_querry = x_querry.sum(dim=1)

        if l not in self.e_layers:
            return None, 0, x_block

        # ---- Get Shared ----
        K_shared = getattr(self, f'shared_k_{l}')
        A_shared = getattr(self, f'shared_a_{l}')
        P_shared = getattr(self, f'shared_p_{l}')

        # ---- Get Private ----
        K = getattr(self, f'private_k_{l}')
        A = getattr(self, f'private_a_{l}')
        p = getattr(self, f'private_p_{l}')
        pt = int(self.private_size / (self.n_tasks))
        s = sum(self.pool_sizes[:self.task_count]) if self.task_count > 0 else 0
        f = s + self.pool_sizes[self.task_count]

        # freeze/control past tasks
        if train:
            if self.task_count > 0:
                K_private = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                A_private = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                P_private = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
            else:
                K_private = K[s:f]
                A_private = A[s:f]
                P_private = p[s:f]
        else:
            K_private = K[0:f]
            A_private = A[0:f]
            P_private = p[0:f]
        # combine
        K = torch.cat([K_shared, K_private], dim=0)
        A = torch.cat([A_shared, A_private], dim=0)
        P = torch.cat([P_shared, P_private], dim=0)

        # ---- attention and cosine sim ----
        a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
        n_K = nn.functional.normalize(K, dim=1)
        q = nn.functional.normalize(a_querry, dim=2)
        aq_k = torch.einsum('bkd,kd->bk', q, n_K)
        P_ = torch.einsum('bk,kld->bld', aq_k, P)

        i = int(self.e_p_length / 2)
        Ek = P_[:, :i, :]
        Ev = P_[:, i:, :]

        # ---- ortho penalty ----
        loss = 0
        if train and self.ortho_mu > 0:
            loss += ortho_penalty(K) * self.ortho_mu
            loss += ortho_penalty(A) * self.ortho_mu
            loss += ortho_penalty(P.view(P.shape[0], -1)) * self.ortho_mu

            if self.use_ddl_loss:
                diversity_loss = self.ddl_loss(P_shared, P_private, self.ddl_angle_threshold)
                loss += self.diversity_lambda * diversity_loss

        return [Ek, Ev], loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class L2Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, ):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(
                            prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            similarity = similarity.t()  # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id,
                                           torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()),
                                                      device=prompt_id.device)])
                    id_counts = torch.cat(
                        [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                major_prompt_id = prompt_id[major_idx]  # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k

            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k

            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            # pdb.set_trace()

            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim

            # pdb.set_trace()
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out