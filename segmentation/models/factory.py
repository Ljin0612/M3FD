from .unet import UNetBaseline


def build_model(config):
    name = config["model"]["name"]
    common = config["model"]
    if name in {"unet_rgb", "unet_ir", "unet_early_fusion"}:
        modality = name.removeprefix("unet_")
        return UNetBaseline(modality, config["num_classes"], common.get("base_channels", 32))
    if name == "univ_seg":
        from .univ_seg import UNIVSegmentation
        return UNIVSegmentation(config["num_classes"], config["data"]["image_size"][0],
                                common.get("checkpoint"), common.get("freeze_encoder", False))
    raise ValueError(f"Unknown model: {name}")
