


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