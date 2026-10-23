"""A frame that will not decode is not the end of the video.

Those two things were treated as one. ``capture.read()`` returning False ended
the clip, so one rejected h264 packet 16 frames into a 288-frame recording
ended the read there. The pipeline then analysed 6% of the file, told the
operator it was "most likely damaged", and built a complete-looking results
page out of the fragment -- which is the dangerous shape of this bug, because
nothing downstream can tell a run over 16 frames from a run over 288.

The file was not damaged. Seeking to frame 17 and carrying on decodes every
one of the remaining frames, which is what :func:`tools.run.batch_session.iter_frames`
now does.
"""
import pytest

from tools.run.batch_session import MAX_CONSECUTIVE_BAD_FRAMES, iter_frames


class FakeCapture:
    """A capture whose reads fail at chosen positions.

    Models the real failure: ``read()`` fails on a bad frame and *keeps*
    failing, because the capture stays parked on the packet it could not
    decode. Only a seek moves it on. Without that detail a test would pass
    against an implementation that merely retried, which does not work.
    """

    def __init__(self, length, bad=(), seekable=True):
        self.length = length
        self.bad = set(bad)
        self.seekable = seekable
        self.pos = 0
        self.seeks = 0

    def read(self):
        if self.pos >= self.length or self.pos in self.bad:
            return False, None
        frame = f"frame-{self.pos}"
        self.pos += 1
        return True, frame

    def set(self, _prop, value):
        if not self.seekable:
            return False
        self.seeks += 1
        self.pos = int(value)
        return True

    def get(self, _prop):
        return self.length


def indices(capture, declared=0):
    return [i for i, _ in iter_frames(capture, declared)]


class TestOneBadFrameDoesNotEndTheVideo:
    def test_the_frames_after_the_bad_one_are_still_read(self):
        got = indices(FakeCapture(288, bad={16}), declared=288)
        assert len(got) == 287
        assert max(got) == 287

    def test_only_the_bad_frame_is_missing(self):
        got = indices(FakeCapture(288, bad={16}), declared=288)
        assert set(range(288)) - set(got) == {16}

    def test_indices_stay_aligned_with_the_real_position(self):
        """The sample-rate stride and every timestamp are derived from this.

        Renumbering after a skip would quietly shift the rest of the
        recording's clock by one frame.
        """
        capture = FakeCapture(60, bad={10, 11})
        for index, frame in iter_frames(capture, 60):
            assert frame == f"frame-{index}"

    def test_a_burst_of_bad_frames_is_stepped_over(self):
        got = indices(FakeCapture(200, bad=set(range(40, 55))), declared=200)
        assert len(got) == 185
        assert 55 in got


class TestItStillStops:
    def test_a_clean_video_reads_to_the_end_and_stops(self):
        assert indices(FakeCapture(50), declared=50) == list(range(50))

    def test_a_clean_video_never_seeks(self):
        """Recovery must cost nothing on a file that does not need it."""
        capture = FakeCapture(50)
        indices(capture, declared=50)
        assert capture.seeks == 0

    def test_a_long_run_of_failures_ends_the_clip(self):
        """Retrying forever on a genuinely truncated file would hang."""
        capture = FakeCapture(10, bad=set(range(10, 400)))
        assert indices(capture, declared=400) == list(range(10))

    def test_it_gives_up_after_the_stated_number_of_tries(self):
        capture = FakeCapture(5, bad=set(range(5, 10_000)))
        indices(capture, declared=0)
        assert capture.seeks < MAX_CONSECUTIVE_BAD_FRAMES + 2

    def test_an_unseekable_capture_ends_rather_than_spinning(self):
        """A stream with no seek support cannot recover, and must not loop."""
        capture = FakeCapture(8, bad={3}, seekable=False)
        assert indices(capture, declared=8) == [0, 1, 2]

    def test_a_wrong_frame_count_does_not_truncate_a_working_file(self):
        """Containers under-report. The real view2191.mp4 declares 288 and
        decodes 317, and capping at the declared count would throw away the
        tail of every such file."""
        got = indices(FakeCapture(317), declared=288)
        assert len(got) == 317


class TestAgainstTheRealFile:
    """The recording this was found on, if it is still in the repository."""

    def test_the_whole_recording_is_recovered(self):
        cv2 = pytest.importorskip("cv2")
        from pathlib import Path

        path = Path("dataset/high-2636(2191-2347)/view2191.mp4")
        if not path.exists():
            pytest.skip("sample recording not present")

        capture = cv2.VideoCapture(str(path))
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        count = sum(1 for _ in iter_frames(capture, declared))
        capture.release()

        # Sequential reads stop at 16 on this file. Anything near that number
        # means the recovery is not working.
        assert count > 250
