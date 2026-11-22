# Video detection pipeline

## 1. Your timings

The first and final runtimes for the full 50 files are unknown. The completion log lists two Parquet files. A third file contributes 9,697 output records, but has no completion entry. The `second` field measures position within a video, so summing it or counting output rows would give no reliable estimate of processing time. A comparison of the two versions requires elapsed times for the same 50 inputs. Identifying the change that saved the most time also requires a measurement with that change isolated.

## 2. Your pipeline

The output shows how a Parquet file maps to individual results. Each video has an identifier made from the Parquet filename and a numeric suffix. Within each video, records begin at second zero and advance one second at a time. Each JSON line contains the video identifier, the second, an `objects` list of detected labels, and a separate integer `people` count. Seconds with no detected objects still receive a line with an empty list. For example, video zero in the first shard has empty lists at seconds 0 through 8, then records `["person"]` and a count of one at second 9. The companion `.done` file lists completed Parquet filenames, one per line. The outputs do not identify the Parquet reader, frame-selection rule, model, or detection thresholds.

## 3. Where the time goes

The records measure output volume. The two shards marked complete produced 9,144 and 9,662 records respectively. Neither file records how long reading, decoding, detection, or writing took, so the bottleneck remains unmeasured. To find it, I would time the full run and each of those stages on the same input. If stages overlap, I would also check which stage makes the others wait. The most informative follow-up would be to change that stage alone and measure the effect on total runtime.

## 4. Checking correctness

I parsed all 28,503 JSON lines and grouped them by video. Each of the 92 videos had consecutive seconds from zero through its final recorded second, with no duplicate timestamps. I also compared the two person fields. Every positive count had a `person` label, but 73 records had that label with a count of zero. One example is second 32 of video zero in the first shard. These checks establish internal consistency for timestamps and flag a counting question. Verifying that optimization preserved detections would require the earlier version’s output for the same video seconds.

## 5. Working with AI

The AI’s first attempt produced a separate results summary. It counted videos, seconds, and detections, but missed the six questions in the required `WRITEUP.md`. I asked it to complete the named file. Comparing the summary with the template exposed a more important gap: the results support checks on timestamps and people counts, while the timing questions require measurements neither output contains. I kept the verifiable counts and left the performance claims open.

## 6. Next steps

With another week, I would first save a baseline for a small, fixed set of videos, including the example at second 32. I would review the 73 label/count disagreements against the corresponding frames and the intended counting rule. That would give me concrete cases to check after each optimization. I would then profile the pipeline, change the slowest stage, and compare both versions on the same 50 files and hardware. For each run I would save elapsed time, completion status, and per-second output so the speedup and any detection changes could be checked together.
