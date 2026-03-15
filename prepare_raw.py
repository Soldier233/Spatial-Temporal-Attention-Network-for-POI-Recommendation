import argparse
import gzip
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


NYC_FILENAME = "dataset_TSMC2014_NYC.txt"
GOWALLA_FILENAME = "loc-gowalla_totalCheckins.txt.gz"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw NYC/Gowalla check-ins into STAN's .npy format.")
    parser.add_argument("--dataset", choices=["NYC", "Gowalla"], required=True)
    parser.add_argument("--raw-dir", default="./data/raw", help="Directory containing downloaded raw datasets.")
    parser.add_argument("--output-dir", default="./data", help="Directory to store processed .npy files.")
    parser.add_argument("--min-poi-freq", type=int, default=None, help="Minimum POI visit count to keep.")
    parser.add_argument("--min-user-checkins", type=int, default=None, help="Minimum user check-ins to keep after POI filtering.")
    parser.add_argument("--top-pois", type=int, default=None, help="Keep only the most frequent POIs after thresholding.")
    return parser.parse_args()


def read_nyc_records(raw_path, min_poi_freq, min_user_checkins):
    poi_counter = Counter()
    rows = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                continue
            user_id, venue_id = parts[0], parts[1]
            lat, lon = float(parts[4]), float(parts[5])
            timezone_offset = int(parts[6])
            utc_dt = datetime.strptime(parts[7], "%a %b %d %H:%M:%S %z %Y")
            local_dt = utc_dt + timedelta(minutes=timezone_offset)
            rows.append((user_id, venue_id, int(local_dt.timestamp() // 60), lat, lon))
            poi_counter[venue_id] += 1

    keep_pois = {poi for poi, freq in poi_counter.items() if freq >= min_poi_freq}
    filtered_rows = [row for row in rows if row[1] in keep_pois]

    if min_user_checkins > 0:
        user_counter = Counter(row[0] for row in filtered_rows)
        keep_users = {user for user, freq in user_counter.items() if freq >= min_user_checkins}
        filtered_rows = [row for row in filtered_rows if row[0] in keep_users]

    return filtered_rows


def collect_gowalla_counts(raw_path):
    poi_counter = Counter()
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            poi_counter[parts[4]] += 1
    return poi_counter


def read_gowalla_records(raw_path, min_poi_freq, min_user_checkins, top_pois):
    poi_counter = collect_gowalla_counts(raw_path)
    keep_pois = [poi for poi, freq in poi_counter.items() if freq >= min_poi_freq]
    keep_pois.sort(key=lambda poi: (-poi_counter[poi], poi))
    if top_pois is not None:
        keep_pois = keep_pois[:top_pois]
    keep_pois = set(keep_pois)

    user_counter = Counter()
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5 or parts[4] not in keep_pois:
                continue
            user_counter[parts[0]] += 1

    keep_users = {user for user, freq in user_counter.items() if freq >= min_user_checkins}

    rows = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            user_id, ts, lat, lon, venue_id = parts
            if venue_id not in keep_pois or user_id not in keep_users:
                continue
            utc_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            rows.append((user_id, venue_id, int(utc_dt.timestamp() // 60), float(lat), float(lon)))

    return rows


def collect_poidata_counts(base_dir):
    poi_counter = Counter()
    for split in ["train.txt", "tune.txt", "test.txt"]:
        with (base_dir / split).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 5:
                    continue
                poi_counter[parts[1]] += 1
    return poi_counter


def read_poidata_records(base_dir, min_poi_freq, min_user_checkins, top_pois):
    poi_counter = collect_poidata_counts(base_dir)
    keep_pois = [poi for poi, freq in poi_counter.items() if freq >= min_poi_freq]
    keep_pois.sort(key=lambda poi: (-poi_counter[poi], poi))
    if top_pois is not None:
        keep_pois = keep_pois[:top_pois]
    keep_pois = set(keep_pois)

    user_counter = Counter()
    for split in ["train.txt", "tune.txt", "test.txt"]:
        with (base_dir / split).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 5 or parts[1] not in keep_pois or "null" in parts[2]:
                    continue
                user_counter[parts[0]] += 1

    keep_users = {user for user, freq in user_counter.items() if freq >= min_user_checkins}
    rows = []
    for split in ["train.txt", "tune.txt", "test.txt"]:
        with (base_dir / split).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 5:
                    continue
                user_id, venue_id, coord, hm, date_id = parts
                if venue_id not in keep_pois or user_id not in keep_users:
                    continue
                if "null" in coord or "," not in coord:
                    continue
                lat, lon = map(float, coord.split(","))
                hour, minute = map(int, hm.split(":"))
                minute_ts = int(date_id) * 24 * 60 + hour * 60 + minute
                rows.append((user_id, venue_id, minute_ts, lat, lon))
    return rows


def remap_rows(rows):
    rows.sort(key=lambda item: (item[0], item[2], item[1]))
    user_vocab = {user_id: idx + 1 for idx, user_id in enumerate(sorted({row[0] for row in rows}))}
    poi_vocab = {poi_id: idx + 1 for idx, poi_id in enumerate(sorted({row[1] for row in rows}))}
    poi_coords = {}
    min_time = min(row[2] for row in rows)

    data = np.zeros((len(rows), 3), dtype=np.int32)
    for idx, (user_id, poi_id, minute_ts, lat, lon) in enumerate(rows):
        mapped_poi = poi_vocab[poi_id]
        data[idx, 0] = user_vocab[user_id]
        data[idx, 1] = mapped_poi
        data[idx, 2] = minute_ts - min_time + 1
        poi_coords.setdefault(mapped_poi, (lat, lon))

    poi = np.zeros((len(poi_vocab), 3), dtype=np.float64)
    for mapped_poi, (lat, lon) in poi_coords.items():
        poi[mapped_poi - 1] = np.array([mapped_poi, lat, lon], dtype=np.float64)

    return data, poi


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "NYC":
        raw_path = raw_dir / NYC_FILENAME
        if not raw_path.exists():
            raise FileNotFoundError(
                f"{raw_path} was not found. The original TSMC NYC source is currently unreliable; "
                "use an existing data/NYC.npy or place dataset_TSMC2014_NYC.txt under data/raw first."
            )
        min_poi_freq = args.min_poi_freq or 10
        min_user_checkins = args.min_user_checkins or 0
        rows = read_nyc_records(raw_path, min_poi_freq=min_poi_freq, min_user_checkins=min_user_checkins)
    else:
        min_poi_freq = args.min_poi_freq or 10
        min_user_checkins = args.min_user_checkins or 10
        top_pois = args.top_pois or 5000
        poidata_dir = raw_dir / "poidata" / "Gowalla"
        raw_path = raw_dir / GOWALLA_FILENAME
        if poidata_dir.exists():
            rows = read_poidata_records(
                poidata_dir,
                min_poi_freq=min_poi_freq,
                min_user_checkins=min_user_checkins,
                top_pois=top_pois,
            )
        else:
            rows = read_gowalla_records(
                raw_path,
                min_poi_freq=min_poi_freq,
                min_user_checkins=min_user_checkins,
                top_pois=top_pois,
            )

    if not rows:
        raise RuntimeError(f"No rows were produced for dataset {args.dataset}.")

    data, poi = remap_rows(rows)
    np.save(output_dir / f"{args.dataset}.npy", data)
    np.save(output_dir / f"{args.dataset}_POI.npy", poi)
    print(
        f"{args.dataset}: users={data[:, 0].max()}, pois={poi.shape[0]}, "
        f"checkins={data.shape[0]}, time_range=({data[:, 2].min()}, {data[:, 2].max()})"
    )


if __name__ == "__main__":
    main()
