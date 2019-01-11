import torch
import torch.nn.functional as F


def kl_divergence(dis1, dis2):
    val = F.kl_div(dis1.log(), dis2, size_average=False, reduce=True)
    return val


def js_divergence(dis1, dis2):
    sym_dis = (dis1 + dis2) / 2
    return 0.5 * kl_divergence(dis1, sym_dis) + 0.5 * kl_divergence(dis2, sym_dis)


def self_make(dis1, dis2):
    to_mul = dis1 / dis2
    return torch.sum(dis1 * torch.log(to_mul))


if __name__ == "__main__":
    p1 = torch.Tensor([0.1, 0.2, 0.1, 0.2, 0.4])
    p2 = torch.Tensor([0.1, 0.2, 0.2, 0.3, 0.3])
    print(kl_divergence(p1, p2))
    print(self_make(p2, p1))
    print(js_divergence(p1, p2))
