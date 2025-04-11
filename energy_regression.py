from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn
from umap import UMAP
from sklearn.cluster import DBSCAN
import joblib
import schnetpack.properties as properties

class Eregression(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        feature_model,
        activation: nn.Module = nn.Softplus(),
        n_clusters: int = 10,
        umap_components: int = 3,
        dbscan_eps: float = 4.2,
        dbscan_min_samples: int = 10,
        aggregation_mode: str = "sum",  # "sum" or "avg"
        output_key: str = "energy_U0",
        enable_smooth: bool = False,  # 是否启用势能面平滑拼接
        task_mode = "train", # "train" or "prediction"

    ):
        super().__init__()
        self.all_molecular_features = []
        self.n_atom_basis = n_atom_basis
        self.n_clusters = n_clusters
        self.umap_components = umap_components
        self.aggregation_mode = aggregation_mode
        self.output_key = output_key
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.enable_smooth = enable_smooth
        self.mode = task_mode
        self.feature_model = feature_model

        self.cluster_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_atom_basis, 16),
                    activation,
                    # nn.Linear(128, 64),
                    # activation,
                    nn.Linear(16, 1),
                )
                for _ in range(n_clusters)
            ]
        )

        self.noise_net = nn.Sequential(
            nn.Linear(n_atom_basis, 16),
            activation,
            nn.Linear(16, 1),
        )

        self.model_outputs = [self.output_key]
        self.best_val_loss = float("inf")  # 初始化为正无穷大

        self.umap = UMAP(n_components=self.umap_components)
        self.dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)

    def forward(self, inputs: Dict[str, torch.Tensor], val_loss: Optional[float] = None):

        inputs = self.feature_model(inputs)
        scalar_representation = inputs["scalar_representation"]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1  # 获取分子数量
        molecular_features = torch.zeros(maxm, scalar_representation.size(-1), device=scalar_representation.device)
        molecular_features = torch.scatter_add(molecular_features, 0,
                                               idx_m.unsqueeze(-1).expand(-1, scalar_representation.size(-1)),
                                               scalar_representation)


        if self.aggregation_mode == "avg":
            molecular_features = molecular_features / inputs[properties.n_atoms].unsqueeze(-1)


        if self.mode == "train":
            with torch.no_grad():
                # molecular_features_np = torch.cat(molecular_features, dim=0).numpy()
                molecular_features_np = molecular_features.detach().cpu().numpy()
                reduced_features_np = self.umap.fit_transform(molecular_features_np)

                cluster_labels_np = self.dbscan.fit_predict(reduced_features_np)

            reduced_features = torch.tensor(reduced_features_np, device=molecular_features.device)
            cluster_labels = torch.tensor(cluster_labels_np, device=molecular_features.device)

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss  # 更新当前最优验证损失
            # 保存 UMAP 和 DBSCAN 模型
                joblib.dump(self.umap, "umap_model.pkl")
                joblib.dump(self.dbscan, "dbscan_model.pkl")


        elif self.mode == "predict":
            # 加载预训练的 UMAP 和 DBSCAN 模型
            self.umap = joblib.load("umap_model.pkl")
            self.dbscan = joblib.load("dbscan_model.pkl")

            with torch.no_grad():
                molecular_features_np = molecular_features.detach().cpu().numpy()
                reduced_features = self.umap.transform(molecular_features_np)
                cluster_labels_np = self.dbscan.fit_predict(reduced_features)

            reduced_features = torch.tensor(reduced_features, device=molecular_features.device)
            cluster_labels = torch.tensor(cluster_labels_np, device=molecular_features.device)




        if self.enable_smooth:
            # 获取聚类中心
            cluster_centers = []
            for cluster_idx in range(cluster_labels.max().item() + 1):
                cluster_mask = cluster_labels == cluster_idx
                if cluster_mask.any():
                    cluster_center = molecular_features[cluster_mask].mean(dim=0)
                    cluster_centers.append(cluster_center)
            cluster_centers = torch.stack(cluster_centers)

            # 计算每个点到各聚类中心的距离
            distances = torch.cdist(molecular_features, cluster_centers)

            # 转换距离为权重（高斯核函数）
            sigma = 1.0  # 标准差，用于控制平滑程度
            weights = torch.exp(-distances**2 / (2 * sigma**2))
            weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化权重

            # 根据权重计算势能
            molecular_energies = torch.zeros(maxm, device=molecular_features.device)
            for cluster_idx, cluster_net in enumerate(self.cluster_nets):
                cluster_energies = cluster_net(molecular_features).squeeze(-1)
                molecular_energies += weights[:, cluster_idx] * cluster_energies


        else:
            molecular_energies = torch.zeros(maxm, device=molecular_features.device)
            for cluster_idx in range(cluster_labels.max().item() + 1):  # 遍历每个有效聚类
                cluster_mask = cluster_labels == cluster_idx
                # cluster_mask_last_20 = cluster_mask[-50:]

                if cluster_mask.any():
                    cluster_features = molecular_features[cluster_mask]
                    cluster_net = self.cluster_nets[cluster_idx]  # 使用对应的 MLP
                    cluster_energies = cluster_net(cluster_features).squeeze(-1)
                    molecular_energies[cluster_mask] = cluster_energies

        noise_mask = cluster_labels == -1
        if noise_mask.any():
            noise_energies = self.noise_net(molecular_features[noise_mask]).squeeze(-1)
            molecular_energies[noise_mask] = noise_energies

        # 将分子能量存储到输入字典
        inputs[self.output_key] = molecular_energies

        return inputs
