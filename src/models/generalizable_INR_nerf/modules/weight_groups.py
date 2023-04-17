import einops
import torch.nn as nn

from .utils import create_params_with_init


class WeightGroups(nn.Module):
    def __init__(self, params_shape_dict, num_groups, weight_dim=768, modulated_layer_idxs=None):
        """
        Args
            params_shape_dict (Dict[torch.Size])
            num_groups (OmegaConfig.ListConfig)
            weight_dim (int)
            modulated_layer_idxs (Optional[Iterable[int]])
        """
        super().__init__()
        assert params_shape_dict

        if len(num_groups) == 1:
            _num_groups = [num_groups[0] for _ in range(len(params_shape_dict))]
        else:
            assert len(params_shape_dict) == len(num_groups)
            _num_groups = num_groups

        if modulated_layer_idxs is None:
            modulated_layer_idxs = list(range(len(params_shape_dict)))
        else:
            assert len(modulated_layer_idxs) > 0

        self.num_group_total = 0
        self.num_groups_dict = dict()
        self.group_idx_dict = dict()
        self.num_vectors_per_group_dict = dict()

        start_idx, end_idx = 0, 0
        for idx, name in enumerate(params_shape_dict):
            if idx not in modulated_layer_idxs:
                continue
            params_shape = params_shape_dict[name]
            num_groups = min(_num_groups[idx], params_shape[1])
            assert params_shape[1] % num_groups == 0

            end_idx = end_idx + num_groups
            self.num_group_total += num_groups
            self.num_groups_dict[name] = num_groups
            self.group_idx_dict[name] = (start_idx, end_idx)
            self.num_vectors_per_group_dict[name] = params_shape[1] // num_groups
            start_idx = end_idx

        weight_groups = create_params_with_init(shape=[self.num_group_total, weight_dim], init_type="normal")
        self.weight_groups = nn.Parameter(weight_groups)

    def get_group_idx_by_name(self, name):
        """in fact unnecessary helper, if you directly access to group_idx_dict"""
        start_idx, end_idx = self.group_idx_dict[name]
        return start_idx, end_idx

    def forward(self, batch_size=None):
        if batch_size is None:
            return self.weight_groups
        else:
            return einops.repeat(self.weight_groups, "n d -> b n d", b=batch_size)
