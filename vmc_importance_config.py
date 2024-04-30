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
    "symmetries_basis": None,
    "spin_inversion": None,
    "J2": 0.99,
    "eval_set_max_size": 50_000,
    "runs": 1,
    "resnet_block_depth": None,
    "resnet_blocks": None,
    "gcnn_additional_generators": None,
    "gcnn_extend_filter1": None,
    "gcnn_filter_size": None,
    "gcnn_channels": None,
    "gcnn_res_blocks": None,
    "cnn_hidden_channels": None,
    "cnn_dilations": None,
    "sign_noise": 0,
}

configs = {
    0: {
        "log_prob_fn": "invariant_cnn",
        "hidden_channels": [32, 32, 32],
        "kernel_size": 2,
    }
}


def get_config(task_id: int):
    return default_config | resolve_config_inheritance(task_id, configs=configs)
