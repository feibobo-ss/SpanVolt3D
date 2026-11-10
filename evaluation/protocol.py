"""Fixed six-class benchmark identifiers; scores are percentages after collection."""
NAMES = ("tower", "conductor", "jumper_conductor", "insulator", "shield_wire", "vegetation")
METRICS = ("mIoU", "mAcc", "OA") + NAMES
SEEDS = (42, 3407, 9173)
MODELS = ("ptv3", "spunet", "ptv1", "ptv2", "oacnns", "litept", "pointnet2")
DOMAINS = ("id", "ood_110", "ood_1000")
