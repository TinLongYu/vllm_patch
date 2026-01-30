from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, get_compressed_expert_map)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import \
    SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import (get_flash_common3_context,
                                               set_flash_common3_context)
from vllm_ascend.ops.fused_moe.experts_selector import (select_experts,
                                                        zero_experts_compute)
from vllm_ascend.ops.fused_moe.moe_comm_method import (AllGatherCommImpl,
                                                       FusedExpertsResult,
                                                       setup_moe_comm_method)
from vllm_ascend.ops.fused_moe.prepare_finalize import QuantType
from vllm_ascend.utils import (AscendDeviceType, enable_sp,
                               get_ascend_device_type, maybe_trans_nz,
                               npu_stream_switch, shared_expert_dp_enabled,
                               shared_experts_calculation_stream)


'''
Patch init for scenario for redundant experts
'''
def __init__(self, *args, **kwargs):
    super(AscendFusedMoE, self).__init__(*args, **kwargs)

    num_experts = kwargs["num_experts"]
    intermediate_size = kwargs["intermediate_size"]

    AscendFusedMoE.moe_counter += 1
    self.moe_instance_id = AscendFusedMoE.moe_counter

    self._expert_map = None
    self.log2phy = None

    if self.quant_config is None:
        self.quant_method = AscendUnquantizedFusedMoEMethod(
            self.moe_config)
    else:
        self.quant_method = self.quant_config.get_quant_method(
            self, self.layer_name)

    assert self.quant_method is not None

    self.moe_config.tp_group = get_tp_group()
    self.moe_config.dp_group = get_dp_group()
    self.moe_config.ep_group = get_ep_group()
    self.moe_config.mc2_group = get_mc2_group()
    self.moe_config.supports_eplb = self.quant_method.supports_eplb
    ascend_config = get_ascend_config()
    # flashcommon3 gate stream
    self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
    if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
        AscendFusedMoE.gate_stream = torch.npu.Stream()
    if self.custom_routing_function is None and self.e_score_correction_bias is not None:
        vllm_config = get_current_vllm_config()
        self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
            dtype=vllm_config.model_config.dtype)

    # init moe
    eplb_config = ascend_config.eplb_config
    self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
        eplb_config, self.moe_instance_id, self.moe_config)
    self.global_num_experts = num_experts + self.global_redundant_expert_num
    self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy
                                                        is not None)
    self.local_num_experts = (torch.sum(
        self._expert_map != -1).item() if self._expert_map is not None else
                                self.global_num_experts)

    # Maybe overwrite
    if int(os.getenv("GLOBAL_NUM_EXPERTS", "")) > 1:
        self.global_num_experts = int(os.getenv("GLOBAL_NUM_EXPERTS", "0"))
        self.num_experts = self.global_num_experts

    if self._expert_map is not None:
        logger.info_once(
            "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
            " number of experts: %s/%s. Experts local to global index map:"
            " %s.", self.ep_rank, self.ep_size, self.local_num_experts,
            self.global_num_experts,
            get_compressed_expert_map(self._expert_map))
    if self.dynamic_eplb:
        self.moe_load = torch.zeros(self.local_num_experts,
                                    dtype=torch.int64).npu()

    self.moe_config.num_experts = self.global_num_experts
    self.moe_config.num_local_experts = self.local_num_experts
    self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

    moe_quant_params = {
        "num_experts": self.local_num_experts,
        "hidden_size": self.hidden_size,
        "intermediate_size_per_partition":
        self.intermediate_size_per_partition,
        "params_dtype": self.params_dtype,
        "weight_loader": self.weight_loader,
    }
    # need full intermediate size pre-sharding for WNA16 act order
    if (self.quant_method.__class__.__name__
            in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
        moe_quant_params["intermediate_size_full"] = intermediate_size
    self.quant_method.create_weights(layer=self, **moe_quant_params)

    self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

    setup_moe_comm_method(self.moe_config)
    self.quant_type = self._get_quant_type()



'''
Need to .clone() for w13 and w2 weights to make it contiguous
'''
def apply(self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            use_grouped_topk: bool,
            top_k: int,
            router_logits: torch.Tensor,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            routed_scaling_factor: float = 1.0,
            e_score_correction_bias: Optional[torch.Tensor] = None,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            enable_force_load_balance: bool = False,
            log2phy: torch.Tensor = None,
            **kwargs) -> torch.Tensor:
    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts)

    if zero_expert_num > 0 and zero_expert_type is not None:
        topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
            expert_indices=topk_ids,
            expert_scales=topk_weights,
            num_experts=global_num_experts,
            zero_expert_type=zero_expert_type,
            hidden_states=x,
        )

    topk_weights = topk_weights.to(x.dtype)
    # this is a naive implementation for experts load balance so as
    # to avoid accumulating too much tokens on a single rank.
    # currently it is only activated when doing profile runs.
    if enable_force_load_balance:
        random_matrix = torch.rand(topk_ids.size(0),
                                    global_num_experts,
                                    device=topk_ids.device)
        topk_ids = torch.argsort(
            random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

    moe_comm_method = get_forward_context().moe_comm_method
    final_hidden_states = moe_comm_method.fused_experts(
        hidden_states=x,
        w1=layer.w13_weight.clone(), ## Clone to make contiguous
        w2=layer.w2_weight.clone(), ## Clone to make contiguous
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        dynamic_eplb=self.dynamic_eplb,
        log2phy=log2phy,
        mc2_mask=kwargs.get("mc2_mask", None))
    if zero_expert_num > 0 and zero_expert_type is not None:
        final_hidden_states += zero_expert_result
    return final_hidden_states
    