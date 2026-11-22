"""Offline regression cases for damaged row groups, time bounds and reference voting."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import reference
import run


class RegressionTests(unittest.TestCase):
    def test_partial_group_preserves_global_row_ids(self):
        parquet = MagicMock()
        parquet.__enter__.return_value = parquet
        parquet.schema_arrow.names = ["mp4"]
        parquet.metadata.num_rows = 5
        parquet.num_row_groups = 2
        parquet.metadata.row_group.side_effect = lambda i: SimpleNamespace(num_rows=(3, 2)[i])

        def batches(*, batch_size, columns, row_groups, use_threads):
            self.assertEqual((batch_size, columns), (1, ["mp4"]))
            batch = MagicMock()
            batch.column.return_value = [SimpleNamespace(as_py=lambda: b"mp4")]
            yield batch
            if row_groups == [0]:
                raise OSError("damaged page")
            yield batch

        parquet.iter_batches.side_effect = batches
        events = []
        status = run.FileStatus()
        with patch.object(run.pq, "ParquetFile", return_value=parquet):
            rows = list(run.iter_videos(Path("shard.parquet"), 4, status, events.append))
        self.assertEqual([key for key, _ in rows], ["shard.parquet:0", "shard.parquet:3"])
        errors = [row for event in events if event[0] == "rows" for row in event[1]]
        self.assertEqual([row["video"] for row in errors], ["shard.parquet:1", "shard.parquet:2"])
        self.assertTrue(all("error" in row for row in errors))
        self.assertFalse(status.complete)  # Fifth row was outside the requested limit.

    def test_duration_cutoff_in_a_timestamp_gap(self):
        stream = MagicMock()
        stream.average_rate = Fraction(25)
        stream.metadata = {}
        stream.codec_context.options = {}
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [stream]
        image = np.zeros((2, 2, 3), np.uint8)
        frames = [
            SimpleNamespace(pts=p, time_base=Fraction(1, 25), width=2, height=2,
                            side_data=[], to_ndarray=lambda format: image)
            for p in (100, 124, 175)
        ]
        container.decode.return_value = iter(frames)
        events = []
        with patch.object(run.av, "open", return_value=container):
            run.decode_video("v", b"mp4", 8, 1, 1, events.append, run.Event(), 2)
        batch, end = events
        self.assertEqual(batch.seconds, [0, 0])
        self.assertEqual((end.last_second, end.error), (1, ""))

    def test_reference_half_vote_median_and_empty_second(self):
        origin = Fraction(29, 7)
        stamps = [origin + t for t in [0, Fraction(999, 1000), 1, 3]]
        frame = np.zeros((2, 2, 3), np.uint8)
        classes = [[0, 0, 2], [0], [], [67]]
        model = MagicMock()
        model.predict.side_effect = [
            [SimpleNamespace(boxes=SimpleNamespace(cls=np.array(items)))] for items in classes
        ]
        events = []
        with patch.object(reference, "timestamps", return_value=(t for t in stamps)), \
                patch.object(reference, "displayed_frames", return_value=(frame for _ in stamps)):
            reference.reference_video("v", b"mp4", model, "cpu", events.append)
        rows = [row for event in events if event[0] == "rows" for row in event[1]]
        self.assertEqual([(r["second"], r["objects"], r["people"]) for r in rows], [
            (0, ["person", "car"], 1), (1, [], 0), (2, [], 0), (3, ["cell phone"], 0)])


if __name__ == "__main__":
    unittest.main()
