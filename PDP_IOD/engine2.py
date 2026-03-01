import os
import sys
import pdb
import tqdm
import torch
from sklearn.cluster import KMeans
import utils
import numpy as np
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets.coco_eval import CocoEvaluator
import torch.nn.functional as F
from datasets.coco_hug import CocoDetection, task_info_coco, create_task_json
from models.image_processing_deformable_detr import DeformableDetrImageProcessor
from models.configuration_deformable_detr import DeformableDetrConfig
from models.modeling_deformable_detr import DeformableDetrForObjectDetection


class local_trainer(pl.LightningModule):
    def __init__(self, train_loader, val_loader, test_dataset, args, local_evaluator, task_id, eval_mode=False):
        super().__init__()

        # 初始化参数（添加缺失参数的默认值）
        if not hasattr(args, 'pseudo_thresh'):
            args.pseudo_thresh = 0.3  # 默认伪标签阈值
        if not hasattr(args, 'use_distillation'):
            args.use_distillation = True  # 默认启用蒸馆
        if not hasattr(args, 'bg_thres_topk'):
            args.bg_thres_topk = 5  # 默认背景阈值 topk

        detr_config = DeformableDetrConfig()
        detr_config.num_labels = args.n_classes  # + 1
        detr_config.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
        detr_config.CUR_INTRODUCED_CLS = args.task_map[task_id][2]
        seen_classes = detr_config.PREV_INTRODUCED_CLS + detr_config.CUR_INTRODUCED_CLS
        self.old_model = None  # 用于知识蒸馏的旧模型
        #### prompt arguments
        detr_config.use_prompts = args.use_prompts
        detr_config.n_tasks = args.n_tasks
        detr_config.num_prompts = args.num_prompts
        detr_config.prompt_len = args.prompt_len
        detr_config.local_query = args.local_query
        detr_config.task_num_classes = args.task_num_classes

        self.invalid_cls_logits = list(range(seen_classes,
                                             args.n_classes - 1))  # unknown class indx will not be included in the invalid class range

        if args.repo_name:
            self.model = DeformableDetrForObjectDetection.from_pretrained(args.repo_name, config=detr_config,
                                                                          ignore_mismatched_sizes=True,
                                                                          default=not (args.mask_gradients),
                                                                          log_file=args.log_file)
            self.processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name)
        else:
            self.model = DeformableDetrForObjectDetection(detr_config, default=not (args.mask_gradients),
                                                          log_file=args.log_file)

            self.processor = DeformableDetrImageProcessor()

        self.task_id = task_id
        self.lr = args.lr
        self.lr_backbone = args.lr_backbone
        self.weight_decay = args.weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.args = args
        self.eval_mode = eval_mode
        self.print_count = 0
        self.evaluator = local_evaluator
        self.evaluator.model = self.model
        self.evaluator.invalid_cls_logits = self.invalid_cls_logits
        self.PREV_INTRODUCED_CLS = args.task_map[task_id][1]

        # -------------------------- 新增代码：类别原型与原型初始化 --------------------------
        self.prototype_cache_capacity = 300  # 每个类别固定样本数
        self.total_classes = args.n_classes - 1  # 排除背景类（背景类索引0~total_classes-1）
        # 1. 类别缓存池：key=类别索引，value=列表（存储该类别的查询向量，最多300个）
        self.class_query_cache = {
            cls_idx: [] for cls_idx in range(self.total_classes)
        }
        #    现在初始化为空字典，因为它将存储 {cls_id: [center1, center2, ...]}
        self.class_prototypes = {}
        # 3. 缓存计数：记录每个类别当前缓存的样本数（避免频繁计算len(cache)）
        self.class_cache_count = {cls_idx: 0 for cls_idx in range(self.total_classes)}

        # 性能优化参数
        self.prototype_update_frequency = 2  # 每10个batch更新一次原型
        self.batch_counter = 0
        self.log_frequency = 5000  # 减少打印频率
        # 新增：只在最后一个epoch更新原型的标志
        self.update_prototypes_last_epoch_only = getattr(args, 'update_prototypes_last_epoch_only', True)  # 默认启用此优化

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def update_class_cache(self, cls_idx, new_query):
        """
        更新单个类别的缓存池：新样本加入，超容量时FIFO替换旧样本（性能优化版本）
        Args:
            cls_idx: 类别索引（int）
            new_query: 新样本的查询向量（Tensor, shape=[d_model]）
        """
        cache = self.class_query_cache[cls_idx]
        # 确保新查询在CPU上存储（节省GPU内存）
        new_query_cpu = new_query.detach().cpu()  # 只使用detach，减少clone开销
        if len(cache) < self.prototype_cache_capacity:
            # 缓存未满：直接添加新样本
            cache.append(new_query_cpu)
            self.class_cache_count[cls_idx] += 1
        else:
            # 缓存已满：FIFO替换（移除第一个元素，添加新元素）
            cache.pop(0)
            cache.append(new_query_cpu)

    def compute_class_prototypes(self, cls_idx):
        """
        基于当前缓存池计算单个类别的原型（均值）（性能优化版本）
        Args:
            cls_idx: 类别索引（int）
        Returns:
            prototype: 类别原型（Tensor, shape=[d_model]）
        """
        cache = self.class_query_cache[cls_idx]
        if len(cache) == 0:
            # 无样本时返回全零向量（确保设备正确）
            return torch.zeros(self.model.config.d_model, device=self.device)
        # 堆叠缓存内的所有查询向量，计算均值
        # 性能优化：直接在GPU上计算，避免重复的设备移动
        device = self.device
        # 使用列表推导一次性移动所有张量到GPU
        cache_on_device = [tensor.to(device, non_blocking=True) for tensor in cache]
        cache_tensor = torch.stack(cache_on_device, dim=0)  # shape=[K, d_model]，K≤300
        prototype = cache_tensor.mean(dim=0)  # shape=[d_model]
        return prototype

    def print_prototype_space_stats(self, prefix=""):
        """
        打印原型空间的统计信息
        Args:
            prefix: 打印前缀，用于区分不同阶段的输出
        """
        print(f"\n{prefix}=== 原型空间统计信息 ===", file=getattr(self.args, 'log_file', None))
        print(f"{prefix}任务ID: {self.task_id}, 前置类别数: {self.PREV_INTRODUCED_CLS}",
              file=getattr(self.args, 'log_file', None))
        print(f"{prefix}总类别数: {self.total_classes}, 缓存容量: {self.prototype_cache_capacity}",
              file=getattr(self.args, 'log_file', None))

        # 统计各类别的缓存情况
        active_classes = []
        empty_classes = []
        total_cached_samples = 0

        for cls_idx in range(self.total_classes):
            cache_count = self.class_cache_count[cls_idx]
            total_cached_samples += cache_count
            if cache_count > 0:
                active_classes.append(cls_idx)
            else:
                empty_classes.append(cls_idx)

        print(f"{prefix}活跃类别数: {len(active_classes)}/{self.total_classes}",
              file=getattr(self.args, 'log_file', None))
        print(f"{prefix}总缓存样本数: {total_cached_samples}", file=getattr(self.args, 'log_file', None))

        # 打印各类别详细信息
        if active_classes:
            print(f"{prefix}活跃类别详情:", file=getattr(self.args, 'log_file', None))
            for cls_idx in active_classes:
                prototype = self.class_prototypes[cls_idx]
                prototype_norm = torch.norm(prototype).item()
                cache_count = self.class_cache_count[cls_idx]
                is_current_task = "当前任务" if cls_idx > self.PREV_INTRODUCED_CLS else "历史任务"
                print(
                    f"{prefix}  类别{cls_idx}: 缓存样本={cache_count}, 原型范数={prototype_norm:.4f}, 任务={is_current_task}",
                    file=getattr(self.args, 'log_file', None))

        print(f"{prefix}========================\n", file=getattr(self.args, 'log_file', None))

    def print_prototype_details(self, cls_idx_list=None, show_vectors=False):
        """
        打印指定类别的详细原型信息
        Args:
            cls_idx_list: 要打印的类别索引列表，None表示打印所有活跃类别
            show_vectors: 是否显示原型向量的前几个维度
        """
        if cls_idx_list is None:
            # 默认打印所有有缓存的类别
            cls_idx_list = [cls_idx for cls_idx in range(self.total_classes)
                            if self.class_cache_count[cls_idx] > 0]

        print(f"\n=== 类别原型详细信息 ===", file=getattr(self.args, 'log_file', None))
        for cls_idx in cls_idx_list:
            if cls_idx >= self.total_classes:
                continue

            prototype = self.class_prototypes[cls_idx]
            cache_count = self.class_cache_count[cls_idx]
            prototype_norm = torch.norm(prototype).item()

            print(f"类别 {cls_idx}:", file=getattr(self.args, 'log_file', None))
            print(f"  缓存样本数: {cache_count}/{self.prototype_cache_capacity}",
                  file=getattr(self.args, 'log_file', None))
            print(f"  原型范数: {prototype_norm:.6f}", file=getattr(self.args, 'log_file', None))
            print(f"  原型设备: {prototype.device}", file=getattr(self.args, 'log_file', None))
            print(f"  原型形状: {prototype.shape}", file=getattr(self.args, 'log_file', None))

            if show_vectors and prototype_norm > 1e-8:
                # 显示原型向量的前10个维度
                vector_preview = prototype[:10].cpu().numpy() if prototype.is_cuda else prototype[:10].numpy()
                print(f"  原型向量预览(前10维): {vector_preview}", file=getattr(self.args, 'log_file', None))

            print("", file=getattr(self.args, 'log_file', None))

    def compute_prototype_similarity(self, feature, cls_label):
        """
        计算特征与类别原型的相似度（性能优化版本）
        Args:
            feature: 查询特征向量 (Tensor, shape=[d_model])
            cls_label: 类别标签 (int)
        Returns:
            similarity: 余弦相似度 (float)
        """
        if cls_label not in self.class_prototypes:
            return 0.0

        prototype = self.class_prototypes[cls_label]
        # 检查原型是否为零向量（未初始化或无样本）
        prototype_norm = torch.norm(prototype)
        if prototype_norm < 1e-8:
            return 0.0
        prototype = prototype.to(feature.device)

        # 计算余弦相似度
        similarity = F.cosine_similarity(feature, prototype, dim=0)
        return similarity.item()

    def update_prototypes_with_clustering(self, num_centers=3):
        """
        在任务训练结束后，使用K-Means聚类为新学习的类别创建多中心原型。
        """
        print(f"\n[Clustering] Starting prototype update with K-Means for Task {self.task_id}...")

        # 只对当前任务新学习的类别进行聚类
        # 注意：这里的范围取决于您如何定义类别ID，通常是之前所有类别到当前所有类别的范围
        start_cls_idx = self.PREV_INTRODUCED_CLS
        end_cls_idx = self.PREV_INTRODUCED_CLS + self.model.config.CUR_INTRODUCED_CLS

        for cls_idx in range(start_cls_idx, end_cls_idx):
            cache = self.class_query_cache.get(cls_idx, [])

            # 确保有足够多的样本进行聚类
            if len(cache) < num_centers:
                if len(cache) > 0:
                    print(f"  - Class {cls_idx}: Only {len(cache)} samples. Using single mean prototype.")
                    # 退回单点原型
                    self.class_prototypes[cls_idx] = [self.compute_class_prototypes(cls_idx)]
                else:
                    print(f"  - Class {cls_idx}: No samples in cache. Skipping.")
                continue

            # 将缓存的特征堆叠成一个张量
            # 聚类在CPU上进行，以避免大量数据在GPU上的开销
            feature_tensor = torch.stack([tensor.cpu() for tensor in cache], dim=0).numpy()

            # 使用 scikit-learn 执行 K-Means 聚类
            kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init='auto')
            kmeans.fit(feature_tensor)

            # 获取聚类中心并存入原型字典，确保它们在正确的设备上
            centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
            self.class_prototypes[cls_idx] = [center for center in centers]

            print(f"  - Class {cls_idx}: Successfully clustered into {num_centers} prototype centers.")

        print("[Clustering] Prototype update finished.\n")

    def generate_old_class_pseudo_labels(self, old_results, labels, old_features=None):
        """
        采用【多中心原型】和【分层筛选】策略生成伪标签:
        1. 高置信度区 (score > high_thresh): 直接作为伪标签。
        2. 中等置信度区 (low_thresh < score <= high_thresh): 与该类别的K个原型中心计算相似度，
           取最大相似度与阈值(0.5)比较，决定是否采纳。
        """
        high_thresh = getattr(self.args, 'pseudo_thresh_high', 0.5)
        low_thresh = getattr(self.args, 'pseudo_thresh_low', 0.2)
        prototype_sim_thresh = getattr(self.args, 'prototype_sim_thresh', 0.5)  # 您设想的0.5阈值
        use_prototype_filtering = getattr(self.args, 'use_prototype_filtering', True)

        for i in range(len(old_results)):
            scores = old_results[i]['scores']
            labels_tensor = old_results[i]['labels']
            boxes = old_results[i]['boxes']

            final_pseudo_boxes = []
            final_pseudo_labels = []

            valid_class_mask = labels_tensor <= self.PREV_INTRODUCED_CLS

            # --- 第一层：处理高置信度样本 (直接采纳) ---
            high_conf_mask = (scores > high_thresh) & valid_class_mask
            if high_conf_mask.any():
                final_pseudo_boxes.append(boxes[high_conf_mask])
                final_pseudo_labels.append(labels_tensor[high_conf_mask])

            # --- 第二层：处理中等置信度样本 (需要多中心原型筛选) ---
            medium_conf_mask = (scores > low_thresh) & (scores <= high_thresh) & valid_class_mask

            if medium_conf_mask.any() and use_prototype_filtering and old_features is not None and hasattr(old_features,
                                                                                                           'last_hidden_state'):

                candidate_indices = torch.nonzero(medium_conf_mask).squeeze(1)
                candidate_labels = labels_tensor[medium_conf_mask]
                candidate_boxes = boxes[medium_conf_mask]

                if candidate_indices.numel() > 0 and candidate_indices.max().item() < \
                        old_features.last_hidden_state.shape[1]:

                    candidate_features = old_features.last_hidden_state[i, candidate_indices, :]

                    unique_labels_in_candidates = torch.unique(candidate_labels)
                    similarity_scores = torch.zeros_like(candidate_labels, dtype=torch.float32)

                    for cls_id in unique_labels_in_candidates:
                        cls_mask = candidate_labels == cls_id

                        # --- 核心修改：与多个原型中心进行比较 ---
                        prototype_centers = self.class_prototypes.get(cls_id.item(), [])

                        if not prototype_centers:
                            continue

                        prototype_tensor = torch.stack(prototype_centers, dim=0).to(candidate_features.device)
                        cls_features = candidate_features[cls_mask]

                        # 批量计算所有特征与所有聚类中心的相似度矩阵 [num_features, K个中心]
                        similarity_matrix = torch.matmul(
                            F.normalize(cls_features),
                            F.normalize(prototype_tensor).T
                        )

                        # 对每个特征，找出它与K个中心相似度中的【最大值】
                        max_similarities, _ = torch.max(similarity_matrix, dim=1)

                        similarity_scores[cls_mask] = max_similarities

                    # 使用最大相似度进行判断：“只要有其中一个大于0.5就属于伪标签”
                    high_sim_mask = similarity_scores >= prototype_sim_thresh

                    if high_sim_mask.any():
                        final_pseudo_boxes.append(candidate_boxes[high_sim_mask])
                        final_pseudo_labels.append(candidate_labels[high_sim_mask])

            # --- 将所有筛选出的伪标签合并到原始标签中 ---
            if final_pseudo_labels:
                all_final_labels = torch.cat(final_pseudo_labels)
                all_final_boxes = torch.cat(final_pseudo_boxes)

                labels[i]['class_labels'] = torch.cat([
                    labels[i]['class_labels'],
                    all_final_labels.to(labels[i]['class_labels'].device)
                ])
                labels[i]['boxes'] = torch.cat([
                    labels[i]['boxes'],
                    all_final_boxes.to(labels[i]['boxes'].device)
                ])

        return labels



    # def generate_old_class_pseudo_labels(self, old_results, labels, old_features=None):
    #     """
	# 	从旧模型的预测中筛选（批量优化）：
	# 	1. 先批量筛选大于置信度阈值且属于之前类别的样本
	# 	2. 统一与原型空间中的之前类别进行相似度匹配
	# 	3. 只保留相似度高于阈值的样本作为伪标签
    #
	# 	Args:
	# 		old_results: 旧模型的检测结果
	# 		labels: 原始标签
	# 		old_features: 旧模型的特征输出 (optional)
	# 	"""
    #     # 性能优化配置
    #     use_prototype_filtering = getattr(self.args, 'use_prototype_filtering', True)
    #     prototype_sim_thresh = getattr(self.args, 'prototype_sim_thresh', 0.3)
    #     # 新增：快速模式，完全跳过原型过滤
    #     fast_mode = getattr(self.args, 'fast_pseudo_mode', False)
    #
    #     # 批量处理每个样本
    #     for i in range(len(old_results)):
    #         scores = old_results[i]['scores']
    #         labels_tensor = old_results[i]['labels']
    #         boxes = old_results[i]['boxes']
    #
    #         # 第一步：批量筛选大于置信度阈值且属于之前类别的样本
    #         valid_score_mask = scores >= self.args.pseudo_thresh
    #         valid_class_mask = labels_tensor <= self.PREV_INTRODUCED_CLS
    #         valid_mask = valid_score_mask & valid_class_mask
    #
    #         if not valid_mask.any():
    #             continue  # 没有符合条件的检测结果
    #
    #         # 提取所有符合基本条件的检测结果
    #         valid_indices = torch.nonzero(valid_mask).squeeze(1)
    #         valid_labels = labels_tensor[valid_mask]
    #         valid_boxes = boxes[valid_mask]
    #
    #         # 第二步：统一与原型空间进行相似度匹配
    #         if (not fast_mode and use_prototype_filtering and old_features is not None and
    #                 hasattr(old_features, 'last_hidden_state') and len(valid_indices) > 0):
    #
    #             try:
    #                 # 确保索引在特征范围内
    #                 max_idx = valid_indices.max().item()
    #                 if max_idx < old_features.last_hidden_state.shape[1]:
    #                     # 批量提取所有候选样本的特征
    #                     candidate_features = old_features.last_hidden_state[i, valid_indices, :]
    #
    #                     # 优化：批量计算相似度，避免逐个循环
    #                     unique_labels = torch.unique(valid_labels)
    #                     similarity_scores = torch.zeros(len(valid_labels), device=valid_labels.device)
    #
    #                     # 按类别批量计算相似度
    #                     for cls_id in unique_labels:
    #                         cls_mask = valid_labels == cls_id
    #                         if cls_id.item() in self.class_prototypes:
    #                             prototype = self.class_prototypes[cls_id.item()].to(candidate_features.device)
    #                             cls_features = candidate_features[cls_mask]  # [n_cls_samples, d_model]
    #                             # 批量计算相似度
    #                             cls_similarities = F.cosine_similarity(cls_features, prototype.unsqueeze(0), dim=1)
    #                             similarity_scores[cls_mask] = cls_similarities
    #
    #                     # 第三步：只保留相似度高于阈值的样本作为伪标签
    #                     high_sim_mask = similarity_scores >= prototype_sim_thresh
    #
    #                     # 最终的伪标签
    #                     final_labels = valid_labels[high_sim_mask]
    #                     final_boxes = valid_boxes[high_sim_mask]
    #                 else:
    #                     # 索引超出范围，跳过原型过滤
    #                     final_labels = valid_labels
    #                     final_boxes = valid_boxes
    #
    #             except Exception as e:
    #                 # 原型匹配失败，使用基本筛选结果
    #                 print(f"Warning: Prototype matching failed: {e}")
    #                 final_labels = valid_labels
    #                 final_boxes = valid_boxes
    #         else:
    #             # 不使用原型过滤，直接使用基本筛选结果
    #             final_labels = valid_labels
    #             final_boxes = valid_boxes
    #         if count_after_threshold > 0:
    #             rejection_rate = 1 - (count_after_prototype / count_after_threshold)
    #             print(
    #                 f"伪标签统计: 阈值后={count_after_threshold}, 原型后={count_after_prototype}, 拒绝率={rejection_rate:.2%}")
    #         # 将筛选后的伪标签添加到原始标签中
    #         if len(final_labels) > 0:
    #             labels[i]['class_labels'] = torch.cat([
    #                 labels[i]['class_labels'],
    #                 final_labels.to(labels[i]['class_labels'].device)
    #             ])
    #             labels[i]['boxes'] = torch.cat([
    #                 labels[i]['boxes'],
    #                 final_boxes.to(labels[i]['boxes'].device)
    #             ])
    #
    #     return labels

    def common_step(self, batch, batch_idx, return_outputs=None):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

        # 安全检查：确保 pre_path 属性存在
        if self.task_id > 1 and not hasattr(self.args, 'pre_path'):
            self.args.pre_path = getattr(self.args, 'output_dir', '').replace(f'Task_{self.task_id}',
                                                                              f'Task_{self.task_id - 1}')

        if hasattr(self.args, 'pre_path') and self.args.pre_path == "/data/zyt/md-detr_prototypes/run/demo_agn/Task_1":
            checkpoint_name = "checkpoint07.pth"
        else:
            checkpoint_name = "checkpoint07.pth"

        # 只在任务ID大于1时构建旧模型路径
        old_model_path = None
        if self.task_id > 1 and hasattr(self.args, 'pre_path'):
            old_model_path = os.path.join(self.args.pre_path, checkpoint_name)
        # elif self.task_id > 1:
        # print("Warning: Cannot determine old model path for task > 1, skipping distillation")

        # 优化：只在第一次加载旧模型，避免重复加载
        old_model_outputs = None  # 存储旧模型输出
        if (not self.eval_mode and self.task_id > 1 and
                hasattr(self.args, 'use_distillation') and self.args.use_distillation and old_model_path):

            # 检查是否已经加载了旧模型
            if not hasattr(self, 'old_model') or self.old_model is None:
                if os.path.exists(old_model_path):
                    self.old_model = deepcopy(self.model)
                    self.old_model.load_state_dict(torch.load(old_model_path)['model'])
                    self.old_model.eval()
                    for param in self.old_model.parameters():
                        param.requires_grad = False
                    print(f"[One-time] Loaded old model from {old_model_path} for distillation")
                else:
                    print(f"Warning: Old model not found at {old_model_path}")
                    self.old_model = None

            # 只在旧模型存在时进行推理
            if self.old_model is not None:
                # 性能优化：在eval模式下使用旧模型推理
                with torch.no_grad():
                    self.old_model.eval()  # 确保在eval模式
                    old_model_outputs = self.old_model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False,
                                                       task_id=self.task_id - 1)
        if old_model_outputs is not None:
            # 旧模型输出后处理为检测结果（框、标签、分数）
            old_results = self.processor.post_process(
                old_model_outputs,
                target_sizes=orig_target_sizes,
                bg_thres_topk=self.args.bg_thres_topk
            )
            # 筛选旧模型中的高置信度旧类别，生成伪标签（新增：传递特征信息）
            labels = self.generate_old_class_pseudo_labels(old_results, labels, old_model_outputs)
        if self.args.use_prompts:
            with torch.no_grad():

                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, train=False,
                                     task_id=self.task_id)

                if not self.args.local_query:
                    query = outputs.last_hidden_state.mean(dim=1)
                else:
                    query = outputs.last_hidden_state
                    outputs_without_aux = {k: v for k, v in outputs.items() if
                                           k != "auxiliary_outputs" and k != "enc_outputs"}
                    # Retrieve the matching between the outputs of the last layer and the targets
                    indices = self.model.matcher(outputs_without_aux, labels)

                    one_hot_proposals = torch.zeros((len(labels), 300)).to(self.device)
                    for i, ind in enumerate(indices):
                        for j in ind[0]:
                            one_hot_proposals[i][j] = 1

                    query_wt = self.model.model.prompts.query_tf(query.view(query.shape[0], -1))
                    query_loss = F.cross_entropy(query_wt, one_hot_proposals)

                if self.args.bg_thres and not return_outputs:
                    results = self.processor.post_process(outputs, target_sizes=orig_target_sizes,
                                                          bg_thres_topk=self.args.bg_thres_topk)

        else:
            query = None

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, query=query, train=True,
                             task_id=self.task_id)

        # -------------------------- 原型空间更新（性能优化版本：只在最后epoch更新） --------------------------
        if self.training and hasattr(outputs, 'last_hidden_state'):
            # 检查是否在最后一个epoch（假设总的epochs保存在args.epochs中）
            is_last_epoch = (self.current_epoch == self.args.epochs - 1) if hasattr(self.args, 'epochs') else False
            if self.update_prototypes_last_epoch_only and is_last_epoch:
                # 只在最后一个epoch进行原型更新
                self.batch_counter += 1
                # 降低更新频率
                if self.batch_counter % self.prototype_update_frequency == 0:
                    # 获取查询向量（decoder的最后一层隐藏状态）
                    query_vectors = outputs.last_hidden_state  # shape: [batch_size, num_queries, d_model]

                    # 获取匹配结果（将预测与真实标签匹配）
                    outputs_without_aux = {k: v for k, v in outputs.items() if
                                           k not in ["auxiliary_outputs", "enc_outputs"]}
                    indices = self.model.matcher(outputs_without_aux, labels)

                    # 批量收集需要更新的原型数据，减少逐个更新的开销
                    prototype_updates = {}  # {cls_label: [query_vectors]}

                    # 遍历每个样本，收集更新数据
                    for batch_idx, (pred_indices, target_indices) in enumerate(indices):
                        current_labels = labels[batch_idx]['class_labels']
                        current_query_vectors = query_vectors[batch_idx]  # [num_queries, d_model]

                        # 遍历每个匹配的预测-标签对
                        for pred_idx, target_idx in zip(pred_indices, target_indices):
                            if target_idx < len(current_labels):
                                cls_label = current_labels[target_idx].item()
                                # 确保类别标签在有效范围内且属于当前任务
                                if 0 <= cls_label < self.total_classes and cls_label > self.PREV_INTRODUCED_CLS:
                                    matched_query = current_query_vectors[pred_idx]  # [d_model]
                                    if cls_label not in prototype_updates:
                                        prototype_updates[cls_label] = []
                                    prototype_updates[cls_label].append(matched_query)

                    # 批量更新原型（减少计算次数）
                    for cls_label, query_list in prototype_updates.items():
                        for query in query_list:
                            self.update_class_cache(cls_label, query)
                        # 在最后epoch时打印更新信息
                        if prototype_updates and self.batch_counter % 100 == 0:
                            print(
                                f"[Last Epoch] Batch {self.batch_counter}: Updated prototypes for {len(prototype_updates)} classes")

            elif not self.update_prototypes_last_epoch_only:
                self.batch_counter += 1
                if self.batch_counter % self.prototype_update_frequency == 0:
                    # 获取查询向量（decoder的最后一层隐藏状态）
                    query_vectors = outputs.last_hidden_state  # shape: [batch_size, num_queries, d_model]
                    outputs_without_aux = {k: v for k, v in outputs.items() if
                                           k not in ["auxiliary_outputs", "enc_outputs"]}
                    indices = self.model.matcher(outputs_without_aux, labels)
                    prototype_updates = {}  # {cls_label: [query_vectors]}
                    for batch_idx, (pred_indices, target_indices) in enumerate(indices):
                        current_labels = labels[batch_idx]['class_labels']
                        current_query_vectors = query_vectors[batch_idx]  # [num_queries, d_model]
                        for pred_idx, target_idx in zip(pred_indices, target_indices):
                            if target_idx < len(current_labels):
                                cls_label = current_labels[target_idx].item()
                                if 0 <= cls_label < self.total_classes and cls_label > self.PREV_INTRODUCED_CLS:
                                    matched_query = current_query_vectors[pred_idx]  # [d_model]
                                    if cls_label not in prototype_updates:
                                        prototype_updates[cls_label] = []
                                    prototype_updates[cls_label].append(matched_query)
                    for cls_label, query_list in prototype_updates.items():
                        for query in query_list:
                            self.update_class_cache(cls_label, query)
                        # 只在有新数据时重新计算原型
                        self.class_prototypes[cls_label] = self.compute_class_prototypes(cls_label)
        # -------------------------- 原型空间更新结束 --------------------------

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        if self.args.local_query:
            loss_dict['query_loss'] = query_loss

            loss += self.args.lambda_query * query_loss

        if return_outputs:

            if self.args.mask_gradients:
                outputs.logits[:, :, self.invalid_cls_logits] = -10e10
                outputs.logits = outputs.logits[:, :, :self.args.n_classes - 1]  # removing background class

            # TODO: fix  processor.post_process_object_detection()
            results = self.processor.post_process(outputs,
                                                  target_sizes=orig_target_sizes)  # convert outputs to COCO api
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}
            res = self.evaluator.prepare_for_coco_detection(res)

            return loss, loss_dict, res

        return loss, loss_dict

    def training_step(self, batch, batch_idx):  # automatic training schedule
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step
        short_map = {'loss_ce': 'ce', 'loss_giou': 'giou', 'cardinality_error': 'car', 'training_loss': 'tr',
                     'loss_bbox': 'bbox', 'query_loss': 'QL'}
        self.log("tr", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(short_map[k], v.item(), prog_bar=True)

        # 性能优化：减少打印频率，但在最后epoch时增加监控
        is_last_epoch = (self.current_epoch == self.args.epochs - 1) if hasattr(self.args, 'epochs') else False
        if is_last_epoch:
            # 最后epoch时更频繁地打印原型空间信息
            if batch_idx % 100 == 0:
                self.print_prototype_space_stats(prefix=f"[Last Epoch {self.current_epoch}, Batch {batch_idx}] ")
        elif batch_idx % 500 == 0:
            # 非最后epoch时保持原来的低频率打印
            self.print_prototype_space_stats(prefix=f"[Epoch {self.current_epoch}, Batch {batch_idx}] ")

        return loss

    def on_after_backward(self, *args):
        # freeze gradients for the classifer weights that do not belong to current task
        for i in range(len(self.model.class_embed)):
            self.model.class_embed[i].weight.grad[:self.PREV_INTRODUCED_CLS, :] = 0
            self.model.class_embed[i].bias.grad[:self.PREV_INTRODUCED_CLS] = 0
        return

    def is_last_epoch(self):
        """ 帮助方法：检查当前是否是最后一个 epoch """
        # 确保 self.args.epochs 存在且 self.current_epoch 从0开始计数
        if not hasattr(self.args, 'epochs'):
            return False
        return self.current_epoch == self.args.epochs - 1

    def on_train_epoch_end(self):
        self.lr_scheduler.step()

        # 性能优化：只在特定条件下保存模型
        should_save = False
        if self.current_epoch and self.current_epoch % self.args.save_epochs == 0:
            should_save = True

        # 最后epoch时一定要保存
        if self.is_last_epoch():
            should_save = True
            # 在最后一个epoch结束时，执行聚类来构建多中心原型
            self.update_prototypes_with_clustering(num_centers=3)

        if should_save:
            print(f"\n[Performance] Saving model at epoch {self.current_epoch}...",
                  file=getattr(self.args, 'log_file', None))
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if start_time:
                start_time.record()

            self.save(self.current_epoch)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
                print(f"[Performance] Model saving completed in {elapsed_time:.2f}s",
                      file=getattr(self.args, 'log_file', None))


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            if not self.eval_mode:
                self.coco_evaluator = CocoEvaluator(self.test_dataset.coco, self.args.iou_types)
            else:
                self.coco_evaluator = self.evaluator.coco_evaluator

        loss, loss_dict, res = self.common_step(batch, batch_idx, return_outputs=True)
        self.coco_evaluator.update(res)

        if batch_idx == self.trainer.num_val_batches[0] - 1:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            stats = self.coco_evaluator.coco_eval[self.args.iou_types[0]].stats

            if self.trainer.global_rank == 0:
                self.evaluator.print_coco_stats(self.current_epoch, stats, self.print_count)
                self.print_count = 1
                if self.args.viz:
                    image_ids = self.evaluator.test_dataset.coco.getImgIds()
                    for id in image_ids[0:self.args.num_imgs_viz]:
                        self.evaluator.vizualize(id=id)

        return loss

    def save(self, epoch):
        print('\n Saving at epoch ', epoch, file=self.args.log_file)

        # 性能优化：简化保存前的统计打印
        if hasattr(self.args, 'verbose_save') and self.args.verbose_save:
            # 只在详细模式下打印原型空间统计
            self.print_prototype_space_stats(prefix=f"[Save Epoch {epoch}] ")

        # 性能优化：使用非阻塞式保存
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'class_query_cache': self.class_query_cache,
            'class_prototypes': self.class_prototypes,
            'class_cache_count': self.class_cache_count,
        }

        # 使用快速保存，减少I/O阻塞
        save_path = os.path.join(self.args.output_dir, f'checkpoint{epoch:02}.pth')
        torch.save(save_dict, save_path)
        print(f'Model saved to {save_path}', file=self.args.log_file)

    def resume(self, load_path=''):
        print('\n Resuming model for task ', self.task_id, ' from : ', load_path, file=self.args.log_file)
        if load_path:
            checkpoint = torch.load(load_path, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)

            # 新增：加载类别原型相关数据（如果存在）
            if 'class_query_cache' in checkpoint:
                self.class_query_cache = checkpoint['class_query_cache']
                print(f"\n Loaded class_query_cache with {len(self.class_query_cache)} classes",
                      file=self.args.log_file)

            if 'class_prototypes' in checkpoint:
                # 加载原型数据（已经在模型设备上）
                self.class_prototypes = checkpoint['class_prototypes']
                print(f"\n Loaded class_prototypes with {len(self.class_prototypes)} classes", file=self.args.log_file)

            if 'class_cache_count' in checkpoint:
                self.class_cache_count = checkpoint['class_cache_count']
                print(f"\n Loaded class_cache_count with {len(self.class_cache_count)} classes",
                      file=self.args.log_file)
            else:
                # 如果没有缓存计数，重新计算
                for cls_idx in range(self.total_classes):
                    self.class_cache_count[cls_idx] = len(self.class_query_cache.get(cls_idx, []))

            # 加载后打印原型空间统计
            self.print_prototype_space_stats(prefix="[Resume] ")

        if not self.args.eval and self.args.freeze:

            freeze = self.args.freeze.split(',')
            for id, (name, params) in enumerate(self.model.named_parameters()):
                params.requires_grad = True
                flag = False
                for n in name.split('.'):
                    if n in freeze:
                        params.requires_grad = False
                        flag = True
                if not flag:
                    print('Trainable ..', name, "  Req grad .. ", params.requires_grad, file=self.args.log_file)

    def match_name_keywords(self, n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    def configure_optimizers(self):
        new_params = self.args.new_params.split(',')

        if self.args.repo_name:
            param_dicts = [
                {"params": [p for n, p in self.named_parameters()
                            if self.match_name_keywords(n, new_params) and p.requires_grad],
                 "lr": self.args.lr,
                 },
                {
                    "params": [p for n, p in self.named_parameters() if
                               not self.match_name_keywords(n, new_params) and p.requires_grad],
                    "lr": self.args.lr_old,
                },
            ]
        else:
            param_dicts = [
                {
                    "params":
                        [p for n, p in self.named_parameters()
                         if
                         not self.match_name_keywords(n, self.args.lr_backbone_names) and not self.match_name_keywords(
                             n, self.args.lr_linear_proj_names) and p.requires_grad],
                    "lr": self.args.lr,
                },
                {
                    "params": [p for n, p in self.named_parameters() if
                               self.match_name_keywords(n, self.args.lr_backbone_names) and p.requires_grad],
                    "lr": self.args.lr_backbone,
                },
                {
                    "params": [p for n, p in self.named_parameters() if
                               self.match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
                    "lr": self.args.lr * self.args.lr_linear_proj_mult,
                }
            ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                           weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)

        return self.optimizer

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader


class Evaluator():
    def __init__(self, processor, test_dataset, test_dataloader, coco_evaluator,
                 task_label2name, args, local_trainer=None, PREV_INTRODUCED_CLS=0,
                 CUR_INTRODUCED_CLS=20, local_eval=0, task_id=0, task_name=None):

        self.processor = processor
        self.local_trainer = local_trainer
        if local_trainer:
            self.model = local_trainer.model
        else:
            self.model = None

        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader
        self.coco_evaluator = coco_evaluator
        self.task_label2name = task_label2name
        self.args = args
        self.local_eval = local_eval
        self.task_id = task_id
        self.task_name = task_name

        # if self.args.mask_gradients:
        prev_intro_cls = PREV_INTRODUCED_CLS
        curr_intro_cls = CUR_INTRODUCED_CLS
        seen_classes = prev_intro_cls + curr_intro_cls
        # self.invalid_cls_logits = list(range(seen_classes, self.args.n_classes-1)) #unknown class indx will not be included in the invalid class range
        self.invalid_cls_logits = list(range(seen_classes,
                                             self.args.n_classes - 1))  # unknown class indx will not be included in the invalid class range

    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = self.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def print_coco_stats(self, epoch, stats, print_count):

        if self.task_name == 'cur':
            task_name = 'Current Task (mAP@C): ' + self.args.task
        elif self.task_name == 'prev':
            task_name = 'Previous Tasks (mAP@P): ' + self.args.task
        else:
            task_name = 'All seen Tasks (mAP@A): ' + self.args.task

        output = [task_name,
                  '\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + '%0.3f' % stats[0],
                  '\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ' + '%0.3f' % stats[1],
                  '\nAverage Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ' + '%0.3f' % stats[2],
                  '\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ' + '%0.3f' % stats[3],
                  '\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ' + '%0.3f' % stats[4],
                  '\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ' + '%0.3f' % stats[5],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ' + '%0.3f' % stats[6],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ' + '%0.3f' % stats[7],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' + '%0.3f' % stats[8],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ' + '%0.3f' % stats[9],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ' + '%0.3f' % stats[10],
                  '\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ' + '%0.3f' % stats[11],
                  '\n\n']

        if print_count == 0:
            print_format = 'w'
        else:
            print_format = 'a'
        with open(self.args.output_dir + '/stats.txt', print_format) as f:
            f.writelines(output)
        f.close()

    def plot_results(self, pil_img, ax, scores, labels, boxes):
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        # plt.figure(figsize=(16,10))
        ax.imshow(pil_img)
        # ax = plt.gca()
        colors = COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=2))
            text = f'{self.task_label2name[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=5,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        ax.grid('off')

    def evaluate(self):
        args = self.args
        model = self.model
        coco_evaluator = self.coco_evaluator
        print("\n Running Final Evaluation... \n", file=self.args.log_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        for idx, batch in enumerate(tqdm.tqdm(self.test_dataloader)):

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in
                      batch["labels"]]  # these are in DETR format, resized + normalized
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

            if self.args.use_prompts:
                # pdb.set_trace()
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False,
                                         task_id=self.task_id)

                    if not self.args.local_query:
                        query = outputs.last_hidden_state.mean(dim=1)
                    else:
                        query = outputs.last_hidden_state
            else:
                query = None

            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, query=query, train=False)

            if self.args.mask_gradients:
                outputs.logits[:, :, self.invalid_cls_logits] = -10e10
                outputs.logits = outputs.logits[:, :, :self.args.n_classes - 1]  # removing background class

            results = self.processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes,
                                                                   threshold=0)  # convert outputs to COCO api
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}
            res = self.prepare_for_coco_detection(res)
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        # if not self.local_trainer or self.local_trainer.trainer.global_rank == 0:
        if self.local_eval:
            self.print_coco_stats(epoch=args.epochs + 1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats,
                                  print_count=1)
        elif self.local_trainer.trainer.global_rank == 0:
            self.print_coco_stats(epoch=args.epochs + 1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats,
                                  print_count=1)

        if args.viz:
            image_ids = self.test_dataset.coco.getImgIds()
            # print(image_ids[0:4])
            for id in image_ids[0:self.args.num_imgs_viz]:
                try:
                    self.vizualize()
                except:
                    continue

    def vizualize(self, id=None, score_threshold=0.18, device='cuda'):
        test_dataset = self.test_dataset
        image_ids = test_dataset.coco.getImgIds()

        if id == None:
            image_id = image_ids[np.random.randint(0, len(image_ids))]
        else:
            image_id = id
        print('Image n°{}'.format(image_id))
        image = test_dataset.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(self.args.test_img_dir, image['file_name']))

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=220)

        # plotting GT
        annotations = test_dataset.coco.imgToAnns[image_id]
        cats = test_dataset.coco.cats
        id2label = {k: v['name'] for k, v in cats.items()}
        scores, labels, boxes = [], [], []
        for annotation in annotations:
            box = annotation['bbox']
            class_idx = annotation['category_id']
            x, y, w, h = tuple(box)
            scores.append(1.0)
            labels.append(class_idx)
            boxes.append((x, y, x + w, y + h))

        ax[0].set_title('GT')
        self.plot_results(image, ax=ax[0], scores=np.array(scores), labels=np.array(labels), boxes=np.array(boxes))

        # plotting model's inference
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        inputs['pixel_mask'] = inputs['pixel_mask'].to(device)

        if self.args.use_prompts:
            with torch.no_grad():
                outputs = self.model(pixel_values=inputs['pixel_values'], pixel_mask=inputs['pixel_mask'], train=False)

                if not self.args.local_query:
                    query = outputs.last_hidden_state.mean(dim=1)
                else:
                    query = outputs.last_hidden_state
        else:
            query = None

        with torch.no_grad():
            outputs = self.model(pixel_values=inputs['pixel_values'], pixel_mask=inputs['pixel_mask'], query=query,
                                 train=False)

        if self.args.mask_gradients:
            outputs.logits[:, :, self.invalid_cls_logits] = -10e10
            outputs.logits = outputs.logits[:, :, :self.args.n_classes - 1]  # removing background class

        # let's only keep predictions with score > 0.3
        task = 'cur'
        if len(self.args.task) > 1:
            task = 'prev'

        results = self.processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]],
                                                               threshold=score_threshold)[0]

        ax[1].set_title('Prediction (Ours)')
        self.plot_results(image, ax=ax[1], scores=results['scores'], labels=results['labels'], boxes=results['boxes'])

        for i in range(2):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()

        plt.savefig(os.path.join(self.args.output_dir, f'{task}_img_{image_id}.jpg'), bbox_inches='tight',
                    pad_inches=0.1)