from dilated_nns_xors import resolve_config_inheritance

default_config = {
    "outer_sample_size": 10_000,
    "inner_sample_size": 10_000,
    "inner_epochs": 1,
    "lr": 1e-3,
    "weight_decay": 0,
    "batch_size": 10_000,
    "max_iter": 15000,
    "lattice": "kagome2x4",
    "use_symmetries": False,
    "use_symmetries.basis": None,
    "spin_inversion": None,
    "J2": 0.99,
    "eval_set_max_size": 50_000,
    "runs": 1,
    "resnet_block_depth": None,
    "resnet_blocks": None,
    "dilations": None,
    "gcnn_additional_generators": None,
    "gcnn_extend_filter1": None,
    "gcnn_filter_size": None,
    "gcnn_channels": None,
    "gcnn_res_blocks": None,
    "cnn_hidden_channels": None,
    "cnn_dilations": None,
    "sign_noise": 0,
    "device": "auto",
}

configs = {
    0: {
        "log_prob_fn": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 2,
        "use_symmetries": True,
        "use_symmetries.basis": "zero_sector",
    },
    1: {
        "_inherit": 0,
        "outer_sample_size": 100_000,
        "inner_epochs": 10,
    },
    2: {
        "lattice": "kagome36round",
        "use_symmetries": True,
        "use_symmetries.basis": "zero_sector",
        "spin_inversion": 1,
        "max_iter": 50000,
        "log_prob_fn": "split_group_res_conv_net",
        "gcnn_additional_generators": ["rotation", "flip"],
        "gcnn_extend_filter1": (1, 1),
        "gcnn_filter_size": (2, 2),
        "gcnn_channels": 16,
        "gcnn_res_blocks": 4,
        "outer_sample_size": 100_000,
        "inner_epochs": 10,
    },
    3: {
        "_inherit": 2,
        "outer_sample_size": 1000_000,
    },
    4: {"_inherit": 2, "outer_sample_size": 10_000, "inner_sample_size": 1000},
}


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)
