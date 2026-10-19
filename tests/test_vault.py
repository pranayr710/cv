"""Tests for at-rest protection of the biometric gallery.

The gallery is children's names beside their face embeddings. Each test is a
property that, if it broke, would look like protection while providing none.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend import vault
from backend.enrollment import EnrolledGallery

cryptography = pytest.importorskip("cryptography")


def _gallery(names=("Asha", "Ben")):
    gallery = EnrolledGallery()
    rng = np.random.default_rng(0)
    for name in names:
        vec = rng.normal(size=512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        gallery.register(name, [vec, vec, vec])
    return gallery


class TestEncryption:
    def test_names_and_embeddings_do_not_survive_on_disk(self, tmp_path,
                                                         monkeypatch):
        """The point of encrypting is that the file stops being readable."""
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        path = tmp_path / "gallery.json"
        _gallery().save(path)

        blob = path.read_text(encoding="utf-8")
        assert "Asha" not in blob
        assert "Ben" not in blob
        assert vault.is_encrypted(json.loads(blob))

    def test_the_right_passphrase_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        path = tmp_path / "gallery.json"
        _gallery().save(path)
        assert [p.name for p in EnrolledGallery.load(path).people] == [
            "Asha", "Ben"]

    def test_a_wrong_passphrase_raises_rather_than_returning_empty(
            self, tmp_path, monkeypatch):
        """An empty gallery would look like a gallery with nobody in it.

        The next enrolment would then save over the real one, destroying the
        records this was meant to protect.
        """
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        path = tmp_path / "gallery.json"
        _gallery().save(path)

        monkeypatch.setenv(vault.KEY_ENV, "the-wrong-one")
        with pytest.raises(ValueError, match="could not be decrypted"):
            EnrolledGallery.load(path)

    def test_a_missing_passphrase_raises_rather_than_returning_empty(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        path = tmp_path / "gallery.json"
        _gallery().save(path)

        monkeypatch.delenv(vault.KEY_ENV)
        with pytest.raises(ValueError, match="encrypted"):
            EnrolledGallery.load(path)

    def test_two_saves_do_not_produce_the_same_ciphertext(self, tmp_path,
                                                          monkeypatch):
        """A fresh salt per save, so identical galleries are not identifiable
        as identical from the file alone."""
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        gallery = _gallery()
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        gallery.save(first)
        gallery.save(second)
        assert (first.read_text(encoding="utf-8")
                != second.read_text(encoding="utf-8"))

    def test_without_a_passphrase_the_gallery_still_works(self, tmp_path,
                                                          monkeypatch):
        """Encryption is opt-in; its absence must not break enrolment."""
        monkeypatch.delenv(vault.KEY_ENV, raising=False)
        path = tmp_path / "gallery.json"
        _gallery().save(path)
        assert [p.name for p in EnrolledGallery.load(path).people] == [
            "Asha", "Ben"]


class TestRetention:
    def test_old_enrolments_are_identified(self):
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        payload = {"people": [{"name": "Old", "enrolled_at": old},
                              {"name": "New", "enrolled_at": new}]}
        stale = vault.expired_people(payload, max_age_days=365)
        assert [p["name"] for p in stale] == ["Old"]

    def test_undated_records_count_as_expired(self):
        """Keeping anything undated forever means the oldest records -- the
        ones a retention policy exists to remove -- are never reached."""
        payload = {"people": [{"name": "Ancient"}]}
        assert vault.expired_people(payload, max_age_days=30)

    def test_a_malformed_date_counts_as_expired(self):
        payload = {"people": [{"name": "Broken", "enrolled_at": "not a date"}]}
        assert vault.expired_people(payload, max_age_days=30)

    def test_registration_dates_the_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv(vault.KEY_ENV, raising=False)
        path = tmp_path / "gallery.json"
        _gallery(("Asha",)).save(path)
        entry = json.loads(path.read_text(encoding="utf-8"))["people"][0]
        assert entry["enrolled_at"]
        datetime.fromisoformat(entry["enrolled_at"])


class TestProtectionIsNotLostByAccident:
    """Every write path has to preserve encryption, not just the encrypt one."""

    def test_a_retention_sweep_leaves_the_gallery_encrypted(self, tmp_path,
                                                            monkeypatch):
        """Running a privacy action destroyed at-rest encryption.

        --expire-after decrypted to filter, then wrote the plaintext result
        because re-encrypting was tied to a separate --encrypt flag. The
        surviving children's names and embeddings landed in the clear.
        """
        import runpy
        import sys

        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        path = tmp_path / "gallery.json"
        _gallery(("Keep",)).save(path)
        assert vault.is_encrypted(json.loads(path.read_text(encoding="utf-8")))

        monkeypatch.setattr(
            sys, "argv",
            ["vault", "--gallery", str(path), "--expire-after", "3650"])
        with pytest.raises(SystemExit):
            runpy.run_module("backend.vault", run_name="__main__")

        after = path.read_text(encoding="utf-8")
        assert vault.is_encrypted(json.loads(after))
        assert "Keep" not in after

    def test_encryption_fails_closed_without_the_library(self, tmp_path,
                                                         monkeypatch):
        """An operator who set the passphrase believes the file is encrypted.

        Writing plaintext with a warning, at a level the CLI does not print,
        leaves them believing it.
        """
        monkeypatch.setenv(vault.KEY_ENV, "a-term-passphrase")
        monkeypatch.setattr(vault, "available", lambda: False)
        with pytest.raises(RuntimeError, match="cannot be encrypted"):
            _gallery(("Asha",)).save(tmp_path / "gallery.json")
