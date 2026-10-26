"""Создание модели. Число входных каналов передаётся снаружи (из списка features)."""
from scenenav.models.unet import UNet


def build_model(cfg: dict, in_channels: int, num_classes: int):
    m = cfg["model"]
    if m["arch"] == "unet":
        return UNet(in_channels, num_classes, m.get("base_channels", 32), m.get("dropout", 0.0))
    if m["arch"] == "smp_unet":
        import segmentation_models_pytorch as smp  # ленивый импорт: нужен только для этого варианта
        return smp.Unet(encoder_name=m.get("encoder", "resnet34"),
                        encoder_weights=m.get("encoder_weights", "imagenet"),
                        in_channels=in_channels, classes=num_classes)
    raise ValueError(m["arch"])
