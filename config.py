from typing import Dict, Optional


def get_specrnet_config(input_channels: int, variant: str = "default", seed: Optional[int] = None) -> Dict:
    """Return a dictionary config for SpecRNet.

    Args:
        input_channels: number of input channels (usually 1)
        variant: one of 'default','no-att','gru1','gap' to alter architecture.
        seed: optional random seed to record in config.
    """
    conf = {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 2,
        # optional flags used by model to toggle components
        "use_attention": True,
        "head": "gru",  # or 'gap'
        "seed": seed,
    }

    if variant in ("gru1", "1layer"):
        conf["nb_gru_layer"] = 1
    if variant in ("no-att", "no_attention"):
        conf["use_attention"] = False
    if variant == "gap":
        conf["use_attention"] = False
        conf["head"] = "gap"

    conf["variant"] = variant
    return conf
