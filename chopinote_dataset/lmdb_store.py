"""LMDB-backed storage for Chopinote-AI token data and annotations.

Single-file replacement for 3.2M+ individual JSON files.
Versioned keys support future re-processing without destroying old data.

Key format:  v{version}:{file_id}:{data_type}
  data_type: tokens | sec | ssf | func | meta | cls | len

Value encoding:
  tokens → raw uint32 LE bytes (struct.pack)
  len    → raw uint64 LE (struct.pack, 8 bytes)
  others → msgpack

Reverse indices stored in separate named databases (idx:level, etc.).

Usage:
    # Training (read-only)
    with LMDBStore.open(path, readonly=True) as db:
        tokens_tensor = db.get_tokens_tensor(file_id)   # torch.Tensor
        ssf = db.get_ssf(file_id)                        # dict

    # Migration (write)
    with LMDBStore.open(path, readonly=False) as db:
        with db.batch_write(5000) as batch:
            for file_id, tokens in ...:
                batch.put(file_id, 'tokens', tokens)
                batch.put(file_id, 'len', len(tokens))

    # Annotation (single writes)
    with LMDBStore.open(path, readonly=False) as db:
        db.put_ssf(file_id, ssf_data)
"""

from __future__ import annotations

import contextlib
import os
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import lmdb
import msgpack
import numpy as np
import torch

# ── Constants ──────────────────────────────────────────────────
VERSION = 4  # Current data version (matches tokens_v4)

# Named database names
DB_MAIN = b'__main__'
DB_LEVEL = b'idx:level'

# Value encoding helpers
def _encode_tokens(tokens: list[int]) -> bytes:
    """Pack token IDs as uint32 LE. 4 bytes per token."""
    return struct.pack(f'<{len(tokens)}I', *tokens)

def _decode_tokens(data: bytes) -> list[int]:
    """Unpack uint32 LE → list[int]. Backward-compat with json.load."""
    return list(struct.unpack(f'<{len(data)//4}I', data))

def _decode_tokens_numpy(data: bytes) -> np.ndarray:
    """Zero-copy numpy view of packed int32 bytes. Caller must .copy() for safety."""
    return np.frombuffer(data, dtype=np.int32)

def _encode_len(length: int) -> bytes:
    return struct.pack('<Q', length)

def _decode_len(data: bytes) -> int:
    return struct.unpack('<Q', data)[0]


class LMDBStore:
    """Versioned LMDB storage for Chopinote-AI data pipeline."""

    def __init__(self, env: lmdb.Environment, main_db, level_db, readonly: bool = True):
        self.env = env
        self.main_db = main_db
        self.level_db = level_db
        self.readonly = readonly

    # ── Factory methods ────────────────────────────────────────

    @classmethod
    def open(
        cls,
        path: str,
        readonly: bool = True,
        map_size: int = 48 * 1024**3,
        max_dbs: int = 10,
    ) -> "LMDBStore":
        """Open an existing LMDB environment."""
        env = lmdb.open(
            path,
            readonly=readonly,
            map_size=map_size,
            max_dbs=max_dbs,
            writemap=not readonly,
            map_async=readonly,
            metasync=not readonly,
            lock=not readonly,
            readahead=readonly,
        )
        main_db = env.open_db(DB_MAIN)
        level_db = env.open_db(DB_LEVEL)
        return cls(env, main_db, level_db, readonly=readonly)

    @classmethod
    def create(
        cls,
        path: str,
        map_size: int = 48 * 1024**3,
        max_dbs: int = 10,
    ) -> "LMDBStore":
        """Create a new LMDB environment. Removes existing if present."""
        import shutil
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        env = lmdb.open(
            path,
            readonly=False,
            map_size=map_size,
            max_dbs=max_dbs,
            writemap=True,
            metasync=False,  # Fast during migration
        )
        main_db = env.open_db(DB_MAIN)
        level_db = env.open_db(DB_LEVEL)
        return cls(env, main_db, level_db, readonly=False)

    def __enter__(self) -> "LMDBStore":
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ── Key helpers ─────────────────────────────────────────────

    @staticmethod
    def _key(file_id: str, data_type: str, version: int = VERSION) -> bytes:
        return f"v{version}:{file_id}:{data_type}".encode('utf-8')

    @staticmethod
    def _level_idx_key(level: int, file_id: str) -> bytes:
        return f"L{level}:{file_id}".encode('utf-8')

    # ── Read methods (DataLoader + annotation) ─────────────────

    def get_raw(self, file_id: str, data_type: str, version: int = VERSION) -> Optional[bytes]:
        """Get raw bytes for any data type."""
        key = self._key(file_id, data_type, version)
        with self.env.begin(db=self.main_db) as txn:
            return txn.get(key)

    def get(self, file_id: str, data_type: str, version: int = VERSION) -> Any:
        """Get and decode any data type. Returns decoded value or None."""
        raw = self.get_raw(file_id, data_type, version)
        if raw is None:
            return None
        if data_type == 'tokens':
            return _decode_tokens(raw)
        if data_type == 'len':
            return _decode_len(raw)
        return msgpack.unpackb(raw, raw=False)

    def get_tokens(self, file_id: str, version: int = VERSION) -> list[int]:
        """Load token IDs as list[int]. Backward-compat with json.load."""
        raw = self.get_raw(file_id, 'tokens', version)
        if raw is None:
            raise FileNotFoundError(f"tokens not found: {file_id}")
        return _decode_tokens(raw)

    def get_tokens_tensor(self, file_id: str, version: int = VERSION) -> torch.Tensor:
        """Load tokens as torch.LongTensor via numpy fast path.

        Avoids Python list[int] allocation (saves ~5KB per sample).
        Uses safe-copy: bytes → numpy.frombuffer → torch.from_numpy.
        """
        raw = self.get_raw(file_id, 'tokens', version)
        if raw is None:
            raise FileNotFoundError(f"tokens not found: {file_id}")
        # numpy frombuffer + torch → no intermediate Python list
        arr = np.frombuffer(raw, dtype=np.int32).copy()
        return torch.from_numpy(arr).long()

    def get_sec(self, file_id: str, version: int = VERSION) -> Optional[dict]:
        return self.get(file_id, 'sec', version)

    def get_ssf(self, file_id: str, version: int = VERSION) -> Optional[dict]:
        return self.get(file_id, 'ssf', version)

    def get_func(self, file_id: str, version: int = VERSION) -> Optional[dict]:
        return self.get(file_id, 'func', version)

    def get_meta(self, file_id: str, version: int = VERSION) -> Optional[dict]:
        return self.get(file_id, 'meta', version)

    def get_cls(self, file_id: str, version: int = VERSION) -> Optional[dict]:
        return self.get(file_id, 'cls', version)

    def get_length(self, file_id: str, version: int = VERSION) -> int:
        raw = self.get_raw(file_id, 'len', version)
        if raw is None:
            return 0
        return _decode_len(raw)

    def has(self, file_id: str, data_type: str, version: int = VERSION) -> bool:
        """Check if a key exists."""
        key = self._key(file_id, data_type, version)
        with self.env.begin(db=self.main_db) as txn:
            return txn.get(key) is not None

    # ── Write methods (migration + annotation) ─────────────────

    def _txn_put(self, txn, file_id: str, data_type: str, value: Any, version: int = VERSION):
        """Put a value within an existing transaction."""
        key = self._key(file_id, data_type, version)
        if data_type == 'tokens':
            raw = _encode_tokens(value) if isinstance(value, list) else value
        elif data_type == 'len':
            raw = _encode_len(value)
        elif isinstance(value, bytes):
            raw = value
        else:
            raw = msgpack.packb(value)
        txn.put(key, raw, overwrite=True)

    def put_tokens(self, file_id: str, tokens: list[int], txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'tokens', tokens, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'tokens', tokens, version)

    def put_sec(self, file_id: str, data: dict, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'sec', data, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'sec', data, version)

    def put_ssf(self, file_id: str, data: dict, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'ssf', data, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'ssf', data, version)

    def put_func(self, file_id: str, data: dict, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'func', data, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'func', data, version)

    def put_meta(self, file_id: str, data: dict, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'meta', data, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'meta', data, version)

    def put_cls(self, file_id: str, data: dict, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'cls', data, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'cls', data, version)

    def put_length(self, file_id: str, length: int, txn=None, version: int = VERSION):
        if txn is not None:
            self._txn_put(txn, file_id, 'len', length, version)
        else:
            with self.env.begin(db=self.main_db, write=True) as t:
                self._txn_put(t, file_id, 'len', length, version)

    # ── Batch write context manager ─────────────────────────────

    @contextmanager
    def batch_write(self, buffer_size: int = 5000) -> "WriteBatch":
        """Context manager for batched writes.

        Commits every buffer_size puts. Use for migration to avoid
        holding too many uncommitted writes in a single transaction.

        Usage:
            with db.batch_write(5000) as batch:
                for file_id, tokens in ...:
                    batch.put(file_id, 'tokens', tokens)
        """
        batch = WriteBatch(self, buffer_size)
        try:
            yield batch
        finally:
            batch.flush()

    # ── Index management ────────────────────────────────────────

    def add_to_index(self, index_db_name: bytes, key: bytes,
                     file_id: str, txn=None):
        """Add a file_id to a reverse index DB."""
        db = self.env.open_db(index_db_name)
        idx_key = key + b':' + file_id.encode('utf-8')
        if txn is not None:
            txn.put(idx_key, b'', db=db, overwrite=True)
        else:
            with self.env.begin(db=db, write=True) as t:
                t.put(idx_key, b'', db=db, overwrite=True)

    def scan_index(self, db_name: bytes, prefix: str) -> Iterator[str]:
        """Cursor scan: yield file_ids matching prefix in an index DB.

        Usage:
            for file_id in db.scan_index(DB_LEVEL, 'L3'):
                ...
        """
        db = self.env.open_db(db_name)
        prefix_bytes = prefix.encode('utf-8')
        with self.env.begin(db=db) as txn:
            cursor = txn.cursor(db=db)
            if cursor.set_range(prefix_bytes):
                for key, _ in cursor:
                    if not key.startswith(prefix_bytes):
                        break
                    # key format: "L3:file_id" → extract file_id
                    yield key.decode('utf-8').split(':', 1)[1]
                cursor.close()

    # ── Cursor-based operations ─────────────────────────────────

    def count_prefix(self, prefix: str, version: int = VERSION) -> int:
        """Count keys with given prefix."""
        prefix_bytes = f"v{version}:{prefix}".encode('utf-8')
        count = 0
        with self.env.begin(db=self.main_db) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix_bytes):
                for key, _ in cursor:
                    if not key.startswith(prefix_bytes):
                        break
                    count += 1
                cursor.close()
        return count

    def iter_file_ids(self, version: int = VERSION) -> Iterator[str]:
        """Yield all unique file_ids in the LMDB."""
        seen = set()
        prefix = f"v{version}:".encode('utf-8')
        with self.env.begin(db=self.main_db) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    parts = key.decode('utf-8').split(':')
                    if len(parts) >= 3:
                        fid = parts[1]
                        if fid not in seen:
                            seen.add(fid)
                            yield fid
                cursor.close()

    def delete_version(self, version: int):
        """Delete all keys for a given version."""
        prefix = f"v{version}:".encode('utf-8')
        with self.env.begin(db=self.main_db, write=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                keys_to_delete = []
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    keys_to_delete.append(key)
                cursor.close()
                for key in keys_to_delete:
                    txn.delete(key)

    # ── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return LMDB environment statistics."""
        stat = self.env.stat()
        with self.env.begin(db=self.main_db) as txn:
            main_entries = txn.stat(self.main_db)['entries']
        with self.env.begin(db=self.level_db) as txn:
            level_entries = txn.stat(self.level_db)['entries']
        return {
            'page_size': stat['psize'],
            'num_entries_main': main_entries,
            'num_entries_level': level_entries,
            'map_size': self.env.info()['map_size'],
            'last_txn_id': self.env.info()['last_txnid'],
        }


class WriteBatch:
    """Buffered batch writer for LMDBStore.

    Accumulates writes in memory, commits in batches of buffer_size.
    """

    def __init__(self, store: LMDBStore, buffer_size: int = 5000):
        self.store = store
        self.buffer_size = buffer_size
        self._buffer: list[tuple[str, str, Any]] = []
        self._txn = None

    def put(self, file_id: str, data_type: str, value: Any, version: int = VERSION):
        """Add a write to the buffer. Flushes when buffer is full."""
        self._buffer.append((file_id, data_type, value, version))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Commit all buffered writes to LMDB."""
        if not self._buffer:
            return
        if self._txn is None:
            self._txn = self.store.env.begin(db=self.store.main_db, write=True)
        try:
            for file_id, data_type, value, version in self._buffer:
                self.store._txn_put(self._txn, file_id, data_type, value, version)
            self._txn.commit()
        except lmdb.MapFullError:
            self._txn.abort()
            self._txn = None
            raise RuntimeError(
                "LMDB map_size exhausted. Re-open with larger map_size "
                "(e.g. map_size=64*1024**3 for 64 GB)."
            )
        except Exception:
            self._txn.abort()
            self._txn = None
            raise
        finally:
            if self._txn is not None:
                self._txn = None
        self._buffer.clear()
