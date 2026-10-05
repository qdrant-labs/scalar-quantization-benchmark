"""Gist-960 recall rerun for the scalar quantization article.

One collection, uploaded once, on Qdrant 1.19.1 (m=16, ef_construct=100, Euclidean,
original vectors on disk). The quantization config is switched in place with
update_collection, so every method runs on identical data:
  1. no quantization
  2. scalar quantization (int8, quantile 0.99), with and without rescoring
  3. 4-bit TurboQuant, with and without rescoring

Ground truth is the dataset's own `neighbors` field (computed by ann-benchmarks),
never anything derived from this script's results. Recall@10 = |returned top-10
intersect true top-10| / 10, which is what the 2023 article calls "precision".

Controls, each of which can fail:
  - exact (brute-force) search must reach recall ~1.0 on a sample: proves the
    point IDs line up with the dataset's neighbor indices.
  - baseline recall must rise with ef: proves the HNSW graph is being searched.
  - baseline recall is compared with the 2023 article's non-quantized numbers.
  - after each switch, the collection config is read back to confirm the
    quantization that is actually active.

Latency is recorded but it comes from a busy laptop, so it is not comparable
to the 2023 tables and is not meant for publication.
"""
import json
import sys
import time

import h5py
import numpy as np
from qdrant_client import QdrantClient, models

NAME = "gist_none"
EFS = [128, 256, 512]
K = 10
BATCH = 500


def wait_green(client, n, label):
    t0 = time.time()
    time.sleep(5)
    while True:
        info = client.get_collection(NAME)
        if (info.status == models.CollectionStatus.GREEN
                and (info.indexed_vectors_count or 0) >= n and info.points_count == n):
            print(f"  [{label}] green, indexed {info.indexed_vectors_count}/{n} after {time.time() - t0:.0f}s", flush=True)
            return info
        time.sleep(10)


def upload(client, train, start_at):
    t0 = time.time()
    for start in range(start_at, len(train), BATCH):
        chunk = np.asarray(train[start:start + BATCH], dtype=np.float32)
        client.upsert(NAME, points=models.Batch(ids=list(range(start, start + len(chunk))), vectors=chunk.tolist()), wait=False)
        if start % 100_000 == 0:
            print(f"  uploaded {start} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  upload done in {time.time() - t0:.0f}s", flush=True)


def recall(client, test, truth, ef, quant_params=None, exact=False, limit_q=None):
    qs = test if limit_q is None else test[:limit_q]
    hits, lat = 0, []
    for i, q in enumerate(qs):
        t0 = time.perf_counter()
        res = client.query_points(
            NAME, query=q.tolist(), limit=K,
            search_params=models.SearchParams(hnsw_ef=ef, exact=exact, quantization=quant_params),
        ).points
        lat.append(time.perf_counter() - t0)
        hits += len({p.id for p in res} & set(truth[i, :K].tolist()))
    return hits / (len(qs) * K), float(np.mean(lat))


def main():
    path = sys.argv[1]
    out_path = path.replace(".hdf5", "-results.json")
    f = h5py.File(path, "r")
    train = f["train"]  # read in slices; 3.8 GB stays on disk
    test = np.asarray(f["test"], dtype=np.float32)
    truth = np.asarray(f["neighbors"])
    n = len(train)
    print(f"train {train.shape} test {test.shape} neighbors {truth.shape}", flush=True)

    client = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=600)
    print("server", client.info().version, flush=True)

    if not client.collection_exists(NAME):
        client.create_collection(
            NAME,
            vectors_config=models.VectorParams(size=train.shape[1], distance=models.Distance.EUCLID, on_disk=True),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
    have = client.get_collection(NAME).points_count
    # Upserts are idempotent: resume a little before the last count to cover
    # any batch that was still in flight when the previous run stopped.
    resume = max(0, (have // BATCH) * BATCH - 10 * BATCH)
    if have < n:
        print(f"resuming upload at {resume} (collection had {have})", flush=True)
        upload(client, train, resume)
    wait_green(client, n, "upload")

    results = {"dataset": "gist-960-euclidean", "qdrant": client.info().version, "k": K,
               "hnsw": {"m": 16, "ef_construct": 100}, "runs": []}

    r, _ = recall(client, test, truth, 128, exact=True, limit_q=50)
    print(f"CONTROL exact recall@10 on 50 queries: {r:.4f}", flush=True)
    results["exact_control"] = r
    if r < 0.99:
        raise SystemExit("exact control failed: IDs do not line up with ground truth")

    phases = [
        ("none", models.Disabled.DISABLED, [("none", None)]),
        ("scalar int8", models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, memory=models.Memory.PINNED)),
         [("no rescore", models.QuantizationSearchParams(rescore=False)),
          ("rescore", models.QuantizationSearchParams(rescore=True, oversampling=1.0))]),
        ("turboquant 4-bit", models.TurboQuantization(turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS4, memory=models.Memory.PINNED)),
         [("no rescore", models.QuantizationSearchParams(rescore=False)),
          ("rescore", models.QuantizationSearchParams(rescore=True, oversampling=1.0))]),
    ]
    for method, qconf, modes in phases:
        print(f"== {method}", flush=True)
        if method != "none":
            client.update_collection(NAME, quantization_config=qconf)
        info = wait_green(client, n, method)
        active = info.config.quantization_config
        print(f"  active quantization: {active}", flush=True)
        for label, qp in modes:
            for ef in EFS:
                r, t = recall(client, test, truth, ef, quant_params=qp)
                print(f"  {method:16s} {label:10s} ef={ef:3d} recall@10={r:.4f} laptop_latency={t * 1000:.1f}ms", flush=True)
                results["runs"].append({"method": method, "mode": label, "ef": ef, "recall_at_10": r,
                                        "laptop_mean_latency_s": t, "active_quantization": str(active)})
        with open(out_path, "w") as out:
            json.dump(results, out, indent=2)

    base = [x["recall_at_10"] for x in results["runs"] if x["method"] == "none"]
    print("CONTROL recall rises with ef:", base == sorted(base) and base[-1] > base[0], base, flush=True)
    print("2023 article non-quantized: [0.802, 0.887, 0.941]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
