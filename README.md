# Per-second YOLO11l video detections

Each input row supplies one MP4. The runner writes one JSON object per video second, including empty seconds. Video identifiers use the shard basename and the zero-based row number within that shard.

## Prepare on Ubuntu

Use Python 3.11 and an NVIDIA driver compatible with the CUDA 12.4 PyTorch wheels. The binary-wheel setup below uses PyAV's bundled FFmpeg libraries and headless OpenCV. It needs no FFmpeg executable, CUDA toolkit, GUI libraries, or additional apt packages on a standard Ubuntu 22.04 Python installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
grep -v '^ultralytics==' requirements.txt | python -m pip install -r /dev/stdin \
    matplotlib pillow pyyaml requests scipy psutil polars ultralytics-thop tqdm
python -m pip install --no-deps ultralytics==8.3.200
python prepare.py
```

The two pip commands install the pinned headless OpenCV wheel. Upstream Ultralytics declares the GUI OpenCV distribution as a dependency, so installing its wheel separately avoids having two distributions own the same `cv2` module. The PyTorch and torchvision pins are a matched CUDA pair.

`prepare.py` fetches the official `yolo11l.pt` using Ultralytics and loads it before placing it beside `run.py`. It verifies an existing local copy on subsequent runs. Move that checkpoint with the project to an offline grader. Keep package installation, weight preparation and dataset acquisition outside the timed run.

## Run

```bash
python run.py --input /data/shards --output results.jsonl
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--input DIR` | required | Read `*.parquet` directly inside this directory, in filename order. |
| `--output PATH` | required | Write JSONL here and file checkpoints to `PATH.done`. |
| `--weights PATH` | `yolo11l.pt` beside the script | Load a local Ultralytics detection checkpoint. Supply the official YOLO11l weights for grading. |
| `--devices LIST` | `0` | Use CUDA indices such as `0,1`, or `cpu`. Indices follow `CUDA_VISIBLE_DEVICES`. |
| `--workers N` | smaller of CPU count and 16 | Maximum concurrent video decoders per device. Lower it for machines with less RAM. |
| `--batch N` | `32` | Maximum frames per prediction call. |
| `--half` / `--no-half` | on for CUDA | Enable FP16 or force FP32. CPU and reference mode use FP32. |
| `--stride N` | `1` | Sample every Nth decoded frame and the first frame of every occupied second. |
| `--reference` | off | Use the first selected device, FP32, batch 1, stride 1, and one synchronous decoder with one codec thread. |
| `--resume` | off | Skip fully checkpointed files and append results. The default starts a fresh output and checkpoint. |
| `--limit-files N` | all | Select the first N remaining sorted shards after applying resume. |
| `--limit-videos N` | all | Select the first N rows per shard. A partially selected shard stays eligible for a full run. |
| `-h` / `--help` | | Print CLI help. |

### Decoding and inference

PyArrow reads only `columns=["mp4"]` in single-row batches. Row-group offsets preserve video identifiers even if reading a group fails. Decoder threads open each MP4 with PyAV, use automatic codec threading, and convert sampled frames to BGR. Display matrices and legacy rotation tags are applied before inference. Quarter turns and reflections use NumPy. Oblique angles use bilinear rotation on the original canvas and can differ slightly at interpolated pixels from FFmpeg.

The first decoded timestamp is the origin. Buckets use the floor of the exact rational elapsed time. A frame missing its timestamp is placed using the number of frames since the last valid timestamp and the average rate. Before the first valid timestamp, this estimate starts from index zero at time zero. If the average rate is also unavailable or nonpositive, the estimate uses 30 fps. A frame timestamp earlier than the origin produces a video error.

Each process warms up the detector once. Its main thread calls Ultralytics with same-shape lists of BGR arrays, image size 640, confidence 0.4, IoU 0.7 and COCO IDs 0, 2, 16, 63 and 67. Ultralytics supplies letterboxing, class-aware NMS and inference mode. The decoder queue holds at most two batches. Each decoder also holds its active video and a pending batch. Batches stop at a shape change, the frame-count limit, or about 192 MiB of BGR pixels. A single large frame can exceed that batch budget. The per-frame pixel ceiling is 8192 × 8192 pixels in total area. Decoder surfaces, compressed videos and model buffers require additional memory.

Multiple devices receive disjoint slices of the sorted file list. One spawned process loads a model on each device with assigned files. Pipes carry results to a single writer. Output order can vary with worker completion. For one large shard, only one GPU has work.

### Aggregation, recovery and progress

Class IDs transfer from the GPU once per batch. Per-frame counts use NumPy `bincount`. An object appears when its detection count is positive in at least half the sampled frames in a second. People counts include zeros and use the median rounded down. Objects retain the order person, car, dog, laptop, cell phone. Timestamp gaps produce an empty object list and zero people.

Results flush as videos finish. A failed video gets one error record and may also have earlier completed seconds. A shard with a bad footer or unreadable row group gets a `<basename>:file` error and remains eligible for retry. A shard whose rows were all visited is complete even when individual videos produced errors. Stderr reports files finished, videos finished, decoded frames per elapsed second and inferred frame counts.

The writer syncs results to disk before checkpointing a shard. When resuming, it removes old rows for the selected unfinished work using a temporary copy of the JSONL in the output directory, then appends replacements. Allow room for this copy. Other shards and rows outside the selected limit are retained. Use one invocation per output path and keep input contents, basenames, weights and inference flags consistent across resumed chunks. Use a different output path for a different experiment.

## Validate against the reference

```bash
python run.py --input /data/shards --output reference.jsonl --reference --limit-files 1 --limit-videos 3
python run.py --input /data/shards --output fast.jsonl --limit-files 1 --limit-videos 3
python evaluate.py --ref reference.jsonl --pred fast.jsonl
```

This is the validation procedure to run on the target machine. No measured accuracy or throughput is included. The reference shares timestamp, orientation and aggregation code with the fast path. The comparison exercises batch inference, scheduling, sampling and precision. FP16 can change borderline confidence or NMS decisions. For a batch-only comparison, add `--no-half` to the fast command. A stride above 1 changes the frame population used for presence and medians.

`evaluate.py` aligns records by video and second and reports micro object-presence F1, exact people agreement, people agreement within one, and error-line counts. A missing second contributes an empty object set and a mismatch to both people metrics. An entirely empty object comparison has F1 1. An empty union of seconds has people agreement 0. PASS requires F1 ≥ 0.95, exact ≥ 85% and within-one ≥ 95%. Exit codes are 0 for PASS, 1 for FAIL and 2 for invalid input. Duplicate second records are invalid.

## Kaggle

Enable a GPU accelerator. Use an internet-enabled preparation session or attach previously prepared weights and wheels as notebook inputs. Kaggle images already have a paired torch and torchvision installation. In a notebook shell, run the following in the project directory to keep those packages. These commands assume the standard Kaggle scientific stack, including NumPy, OpenCV, PyArrow, PyYAML, Matplotlib, SciPy, Pillow, psutil, Polars and requests.

```bash
python -m pip install --no-deps ultralytics==8.3.200 ultralytics-thop==2.0.17 av==16.0.1
python prepare.py
```

For an attached checkpoint, pass its path with `--weights`. Kaggle's preinstalled package versions can differ from the grader pins.

The assignment uses shards 00000 through 00049 from HuggingFaceFV/finevideo. Sign in as the owner of the read token, accept the dataset terms, and enable the `HF_TOKEN` secret for the Kaggle notebook. A fine-grained token needs permission to read the gated dataset. [FineVideo access page](https://huggingface.co/datasets/HuggingFaceFV/finevideo)

After cloning the project, run these Kaggle notebook commands from its directory. Each command can run in its own cell. Start with the access check, which prints the token owner's username and checks file metadata on the first shard before downloading any data.

```python
!python kaggle_run.py access
!python kaggle_run.py setup
!python kaggle_run.py sample --videos 1 --minutes 15
!python kaggle_run.py next --minutes 60
!python kaggle_run.py status
```

The setup command performs the installation and weight preparation described above. The sample command times fast and reference inference on the same rows, allows 15 minutes for each run, and saves both JSONL files plus the evaluator output in `/kaggle/working/takehome-results/summary-1.json`. The next command downloads one shard at a time and appends to `results.jsonl` in that directory. Repeat it for additional full shards. The minute caps apply to inference and exclude downloads. After a timeout, finished videos may already have output; the next invocation retries the unfinished shard. Save the JSONL and its matching .done file together across Kaggle sessions. Report the measured sample and completed shard count when submitting partial results.

To download all assigned files during a separate preparation session using the pinned Hub client:

```bash
huggingface-cli download HuggingFaceFV/finevideo --repo-type dataset \
    --include 'data/train-000[0-4]*.parquet' --local-dir /kaggle/working/dataset
```

Set `SHARDS` to the directory that directly contains the Parquet files. Keep output and sidecar in writable storage. To process five new files each time, repeat the same command until stderr reports no remaining files.

```bash
SHARDS=/kaggle/working/dataset/data
python run.py --input "$SHARDS" --output /kaggle/working/results.jsonl --limit-files 5 --resume
```

Across Kaggle sessions, save **both** `results.jsonl` and `results.jsonl.done` and restore them to the same writable location. For a GPU with less memory, lower `--batch`. For host RAM pressure, lower `--workers`.
