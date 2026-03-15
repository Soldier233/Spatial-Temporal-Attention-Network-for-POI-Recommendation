import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

max_len = 100  # max traj len; i.e., M


def haversine_matrix(coords_a, coords_b):
    lat1 = np.radians(coords_a[:, 0])[:, None]
    lon1 = np.radians(coords_a[:, 1])[:, None]
    lat2 = np.radians(coords_b[:, 0])[None, :]
    lon2 = np.radians(coords_b[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    term = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2 * 6371.0 * np.arcsin(np.sqrt(term))).astype(np.float32)


def rst_mat1(traj, poi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    loc_ids = traj[:, 1].astype(np.int64) - 1
    coords = poi[loc_ids][:, 1:3]
    dist = haversine_matrix(coords, coords)
    delta_t = np.abs(traj[:, 2][:, None] - traj[:, 2][None, :]).astype(np.float32)
    return np.stack([dist, delta_t], axis=-1)


def rs_mat2s(poi, chunk_size=256):
    coords = poi[:, 1:3]
    loc_max = len(coords)
    mat = np.zeros((loc_max, loc_max), dtype=np.float32)
    for start in range(0, loc_max, chunk_size):
        end = min(start + chunk_size, loc_max)
        print(start) if start % 1000 == 0 else None
        mat[start:end] = haversine_matrix(coords[start:end], coords)
    return mat


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    hist = traj_time[:-1]
    label = traj_time[1:]
    return np.abs(label[:, None] - hist[None, :]).astype(np.float32)


def process_traj(dname, data_dir="./data"):
    # data (?, [u, l, t]), poi (L, [l, lat, lon])
    base_dir = Path(data_dir)
    data = np.load(base_dir / f"{dname}.npy").astype(np.int64)
    poi = np.load(base_dir / f"{dname}_POI.npy")
    num_user = int(np.max(data[:, 0]))
    data_user = data[:, 0]
    trajs, labels, mat1, mat2t, lens = [], [], [], [], []
    u_max, l_max = int(np.max(data[:, 0])), int(np.max(data[:, 1]))

    for u_id in range(1, num_user + 1):
        init_mat1 = np.zeros((max_len, max_len, 2), dtype=np.float32)
        init_mat2t = np.zeros((max_len, max_len), dtype=np.float32)
        user_traj = data[np.where(data_user == u_id)]
        user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()

        if len(user_traj) < 4:
            continue

        if u_id % 100 == 0:
            print(u_id, len(user_traj))

        if len(user_traj) > max_len + 1:
            user_traj = user_traj[-max_len - 1 :]

        user_len = len(user_traj[:-1])
        user_mat1 = rst_mat1(user_traj[:-1], poi)
        user_mat2t = rt_mat2t(user_traj[:, 2])
        init_mat1[0:user_len, 0:user_len] = user_mat1
        init_mat2t[0:user_len, 0:user_len] = user_mat2t

        trajs.append(torch.as_tensor(user_traj[:-1], dtype=torch.long))
        mat1.append(init_mat1)
        mat2t.append(init_mat2t)
        labels.append(torch.as_tensor(user_traj[1:, 1], dtype=torch.long))
        lens.append(user_len - 2)

    mat2s = rs_mat2s(poi)
    zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
    trajs, mat1, mat2t, labels, lens = zipped
    trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
    trajs = pad_sequence(trajs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    processed = [trajs, np.array(mat1), mat2s, np.array(mat2t), labels, np.array(lens), u_max, l_max]
    data_pkl = base_dir / f"{dname}_data.pkl"
    with data_pkl.open("wb") as pkl:
        joblib.dump(processed, pkl)
    print(f"Saved processed dataset to {data_pkl}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare STAN trajectory tensors from .npy files.")
    parser.add_argument("--dataset", default="NYC", help="Dataset prefix under ./data, for example NYC or Gowalla.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing <dataset>.npy and <dataset>_POI.npy.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_traj(args.dataset, data_dir=args.data_dir)
