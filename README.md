# Scalar Quantization vs 4-bit TurboQuant on Gist-960

Reproduction kit for the "Where Scalar Quantization Fits Today" section of
[Scalar Quantization for Vector Search](https://qdrant.tech/articles/scalar-quantization/).

It measures recall@10 for no quantization, scalar quantization (int8), and
4-bit TurboQuant, each with and without rescoring, at `hnsw_ef` 128, 256, and 512.

## Setup

| Item | Value |
|---|---|
| Dataset | [Gist-960](http://ann-benchmarks.com/gist-960-euclidean.hdf5) from ann-benchmarks: 1,000,000 train vectors, 1,000 queries, 960 dimensions, Euclidean |
| Ground truth | The dataset's own `neighbors` field |
| Qdrant | v1.19.2 (published table) and v1.19.1 (earlier run) |
| Client | `qdrant-client` 1.19.1, Python 3.14 |
| HNSW | `m=16`, `ef_construct=100` |
| Storage | Original vectors on disk, quantized vectors pinned in RAM |
| Scalar quantization | `int8`, `quantile=0.99` |
| TurboQuant | `bits4` |
| Rescoring | `rescore=True`, `oversampling=1.0`, or `rescore=False` |
| Date | October 5, 2026 |

All methods run on one collection, uploaded once. The quantization config is
switched in place with `update_collection`, so every method sees identical data.

## Run

```bash
# Qdrant on localhost:6333, for example:
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.2

pip install qdrant-client==1.19.1 h5py numpy
curl -O http://ann-benchmarks.com/gist-960-euclidean.hdf5
python bench_sq.py gist-960-euclidean.hdf5
```

The upload uses gRPC and takes a few minutes. The full run takes about 20 minutes
on a laptop.

## Controls

The script stops or reports if a control fails:

- Exact search must reach recall@10 of at least 0.99 on 50 queries, which proves
  point IDs line up with the ground truth. Both runs reached 1.000.
- Baseline recall must rise with `hnsw_ef`, which proves the HNSW graph is searched.
- The active quantization config is read back after each switch.

## Results

Raw results for both runs are in `results/`. Individual values moved by up to
0.008 between the two runs, which is the variation between HNSW builds, so
differences smaller than that are not meaningful. Without rescoring, scalar
quantization kept 0.009 to 0.016 more recall than 4-bit TurboQuant in both runs.

Latency in the result files comes from a busy laptop and is not comparable
across machines. The article does not publish it.
