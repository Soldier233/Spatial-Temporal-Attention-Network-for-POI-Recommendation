import torch
from torch import nn
from torch.nn import functional as F

from load import max_len

seed = 0
global_seed = 0
hours = 24 * 7
torch.manual_seed(seed)


def to_npy(x):
    return x.detach().cpu().numpy()


def resolve_device(device_name="auto"):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    target = torch.device(device_name)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this environment.")
    return target


class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(max_len, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max

    def forward(self, self_attn, self_delta, traj_len):
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        device = self_attn.device
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)
        n_batch, loc_count, _ = self_delta.shape
        candidates = torch.arange(1, self.loc_max + 1, device=device, dtype=torch.long)
        candidates = candidates.unsqueeze(0).expand(n_batch, -1)
        emb_candidates = self.emb_loc(candidates)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)
        attn_out = self.value(attn).view(n_batch, loc_count)
        return attn_out


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            valid_len = int(traj_len[i].item())
            mask[i, 0:valid_len, 0:valid_len] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)
        attn = F.softmax(attn, dim=-1) * mask
        return torch.bmm(attn, self.value(joint))


class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super().__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = [float(item) for item in ex]
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]):
            valid_len = int(traj_len[i].item())
            mask[i, 0:valid_len] = 1
            valid_loc = (traj_loc[i] - 1)[:valid_len]
            delta_s[i, :valid_len] = torch.index_select(mat2, 0, valid_loc)

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / max(self.su - self.sl, 1e-6)
        time_interval = (etl * vtu + etu * vtl) / max(self.tu - self.tl, 1e-6)
        return space_interval + time_interval


class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super().__init__()
        self.emb_t, self.emb_l, self.emb_u, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = [float(item) for item in ex]
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        traj = traj.clone()
        traj[:, :, 2] = (traj[:, :, 2] - 1) % hours + 1
        time = self.emb_t(traj[:, :, 2])
        loc = self.emb_l(traj[:, :, 1])
        user = self.emb_u(traj[:, :, 0])
        joint = time + loc + user

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            valid_len = int(traj_len[i].item())
            mask[i, 0:valid_len, 0:valid_len] = 1

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vsu = (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtl = (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)
        vtu = (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / max(self.su - self.sl, 1e-6)
        time_interval = (etl * vtu + etu * vtl) / max(self.tu - self.tl, 1e-6)
        return joint, space_interval + time_interval
