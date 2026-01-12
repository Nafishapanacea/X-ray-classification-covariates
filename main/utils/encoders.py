import torch

def encode_view(frontal_lateral, ap_pa):
    # 0: PA, 1: AP, 2: Lateral
    if frontal_lateral == "lateral":
        return 2
    if ap_pa == "PA":
        return 0
    return 1

def encode_sex(sex):
    return 0 if sex == "M" else 1