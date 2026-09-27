# Forward ledger schema v1

SQLite is the authoritative store (`data/forward.sqlite`). CSV remains a read-only legacy archive. SQLite was selected over multiple append-only files because one transaction can atomically persist a snapshot and prediction, or both sources and a verification. No external database is required.

Every table has `id TEXT PRIMARY KEY`, canonical JSON `payload`, and a SHA-256 `hash`. Tables: `experiments`, `snapshots`, `predictions`, `observations`, `verifications`, `events`.

Predictions additionally have foreign keys to experiment and snapshot, and a unique `(experiment_id, origin)` constraint. Verifications have a prediction foreign key, a checked integer horizon 1–24, and a unique `(prediction_id, horizon)` constraint. Their unique constraints create indexes for these access paths.

`foreign_keys=ON`, `synchronous=FULL`, rollback-journal transactions, and schema `user_version=1` are mandatory. UPDATE and DELETE triggers reject mutation on all tables. Duplicate insertion fails; command-level reruns explicitly skip existing origins or verified horizons. Startup checks SQLite integrity, foreign keys, payload hashes and prediction/snapshot semantics. An interrupted uncommitted process rolls back on reopen.

Canonical JSON uses sorted keys, compact separators and `allow_nan=False`. Input hash covers timestamped raw candles, exact input float32 values, dtype and series definition. Forecast hash covers 24 explicit target points and q10/q90. Verification never updates prediction data.

`data/forward-export.json` is a readable derived export. Reports and exports use temporary-file write, fsync, replace, and parent-directory fsync. GitHub concurrency serializes writers; a failed push is an operational failure, never a successful persistence claim.

Immutability is an application/database guarantee, not protection against a privileged actor deleting the database or rewriting both payloads and hashes. Git history adds provenance but is not a third-party timestamp attestation. SQLite is not encrypted; all records here are public market/research data.
