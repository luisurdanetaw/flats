//! The live-row cursor — the scan primitive the VM executes `SeekFirst` /
//! `Next` / `Column` against.
//!
//! Storage ships the two halves of a scan but never composed them: the metadata
//! index says WHICH ordinals are live
//! ([`meta::Reader::live`](crate::metadata::index::Reader::live)) and the tuple
//! store fetches columns for ONE KNOWN ordinal ([`tuples::Reader::get`]). This
//! is the iterator that sweeps the first and feeds the second.
//!
//! # The ordinal source is a seam, deliberately
//!
//! A cursor is built over *any* iterator of ordinals — it is NOT wired to
//! `live()`. [`Db::scan`](crate::Db::scan) happens to feed it `live()` for a
//! full scan; a `WHERE`-filtered bitmap (`lookup_range(..) & live`) or a ranked
//! KNN result feeds the SAME cursor later, with no new type. That is the whole
//! reason `Cursor::over` takes an iterator instead of a `&MetadataIndex`.
//!
//! # Skip policy: `Missing`/`Deleted` are transients, never errors
//!
//! An enumerated ordinal can fail to produce a row for two ROUTINE reasons, and
//! the cursor skips both:
//!
//!   * [`RowGet::Missing`] — the row is mid-apply. The engine's fan-out writes
//!     flat → meta → tuple across three independent mutexes, so a reader can
//!     catch an ordinal already published into the metadata index whose tuple
//!     has not landed yet. `tests/concurrent_stores.rs` fires this window
//!     ~20-28k times in ~6s of writing; its `WORLD VERDICT` header is the
//!     evidence behind this policy — READ IT before changing anything here.
//!     Skipping is not a fudge: apply runs BEFORE the ack (`wal.rs`,
//!     `commit_batch`), so such a row belongs to an insert whose ack has not
//!     reached its caller. No query owes visibility to a write that has not
//!     returned to the writer.
//!   * [`RowGet::Deleted`] — deleted after the ordinal source was snapshotted.
//!     The symmetric case, equally routine.
//!
//! Turning either into an error would be a once-in-millions production panic:
//! an ascending walk reaches the frontier long after the window usually shuts,
//! so the full-scan pattern hit `Missing` just ONCE in ~3.1M gets on one seed
//! and zero on the next — rare enough to pass casual testing, never rare enough
//! to be safe. The skip lives inside [`Cursor::seek_first`] / [`Cursor::next`],
//! so the VM's `Column` only ever sees a materialized row.
//!
//! # One `get` per row, not per column
//!
//! Advancing materializes the whole row and caches it, so [`Cursor::column`] is
//! an infallible in-memory index. Two reasons beyond the obvious one lock
//! instead of N: `Op::Column` wants an infallible read, and a re-fetching
//! `column` could see the row deleted mid-row and hand back a torn projection.

use crate::error::Result;
use crate::metadata::common::{ColumnId, Ordinal, Value};
use crate::metadata::tuples::{self, RowGet};

/// A forward-only cursor over a sequence of ordinals, yielding the rows that
/// are actually materialized in the tuple store.
///
/// Positioned BEFORE the first row on construction: call
/// [`seek_first`](Self::seek_first) to land on a row, then
/// [`next`](Self::next) to advance. Both report whether a row is available;
/// while one is, [`row`](Self::row) / [`column`](Self::column) read it.
///
/// The lifetime is the ORDINAL SOURCE's, not the reader's: the tuple handle is
/// a cheap `Arc` clone the cursor owns, so a `Db::scan` cursor borrows nothing
/// and is `Cursor<'static>`. Only a caller that iterates a *borrowed* bitmap
/// ties the cursor to a shorter life.
pub struct Cursor<'a> {
    /// The ordinals to visit, in the source's order (ascending for a bitmap).
    ordinals: Box<dyn Iterator<Item = Ordinal> + 'a>,
    /// Read handle into the tuple store. OWNED (an `Arc` clone), not borrowed:
    /// `Db::scan` reaches the reader through a temporary `Arc<Collection>` from
    /// the catalog snapshot, so there is no `&Reader` outliving the call to
    /// hand out. Cloning costs one atomic increment.
    tuples: tuples::Reader,
    /// The columns materialized per row, in the order [`row`](Self::row)
    /// returns them. THE PROJECTION, fixed at construction.
    columns: Vec<ColumnId>,
    /// The ordinal the cursor is parked on, or `None` before the first
    /// `seek_first` and after exhaustion.
    current: Option<Ordinal>,
    /// The current row's values, cached by the advance that landed on it.
    row: Option<Vec<Value>>,
    /// How many ordinals were skipped as `Missing`/`Deleted`. Diagnostics only
    /// — a healthy scan under concurrent writes reports a nonzero count.
    skipped: u64,
    /// How many rows were actually materialized out of the tuple store.
    fetched: u64,
}

impl<'a> Cursor<'a> {
    /// Build a cursor over an arbitrary ordinal source.
    ///
    /// `columns` is the projection: the storage [`ColumnId`]s to materialize
    /// per row, in the order [`row`](Self::row) hands them back. Duplicates and
    /// reordering are fine — the tuple store honors the request order.
    ///
    /// This is the constructor both [`Db::scan`](crate::Db::scan) and any
    /// future filtered/KNN path use; there is exactly one cursor type.
    pub fn over(
        ordinals: impl Iterator<Item = Ordinal> + 'a,
        tuples: tuples::Reader,
        columns: Vec<ColumnId>,
    ) -> Cursor<'a> {
        Cursor {
            ordinals: Box::new(ordinals),
            tuples,
            columns,
            current: None,
            row: None,
            skipped: 0,
            fetched: 0,
        }
    }

    /// Move to the first row. `false` means the scan is empty — the VM's
    /// `SeekFirst` jumps past the loop body on `false`.
    ///
    /// Skips any leading ordinal whose row is not materialized (see the module
    /// header): an empty source and a source of nothing but transients are
    /// indistinguishable to the caller, by design.
    pub fn seek_first(&mut self) -> Result<bool> {
        self.advance()
    }

    /// Advance to the next row. `false` means the scan is exhausted — the VM's
    /// `Next` falls through past the loop on `false`.
    // Deliberately not `Iterator::next`: the cursor hands out a BORROW of the
    // current row (`row`/`column`), which `Iterator` cannot express, and the
    // VM's opcode is a positioning command rather than a value producer.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<bool> {
        self.advance()
    }

    /// Pull ordinals until one yields a real row. The ONLY place the skip
    /// policy lives — `Missing`/`Deleted` are consumed here so no caller can
    /// observe them, let alone mistake one for corruption.
    fn advance(&mut self) -> Result<bool> {
        for ordinal in self.ordinals.by_ref() {
            // `get` errs only on a ColumnId outside the schema — a caller bug
            // in the projection, not a transient, so it propagates.
            match self.tuples.get(ordinal, &self.columns)? {
                RowGet::Live(values) => {
                    self.current = Some(ordinal);
                    self.row = Some(values);
                    self.fetched += 1;
                    return Ok(true);
                }
                // World A: both are routine. Skip, count, keep going.
                RowGet::Missing | RowGet::Deleted => {
                    self.skipped += 1;
                }
            }
        }
        self.current = None;
        self.row = None;
        Ok(false)
    }

    /// The ordinal the cursor is parked on, or `None` if it is not on a row.
    pub fn ordinal(&self) -> Option<Ordinal> {
        self.current
    }

    /// The current row's values, in the projection's column order, or `None`
    /// if the cursor is not on a row.
    pub fn row(&self) -> Option<&[Value]> {
        self.row.as_deref()
    }

    /// One value of the current row by POSITION IN THE PROJECTION — i.e. an
    /// index into [`row`](Self::row), not a [`ColumnId`].
    ///
    /// For a [`Db::scan`](crate::Db::scan) cursor the two coincide: that
    /// projection is every scalar column in `ColumnId` order, so position `i`
    /// IS `ColumnId(i)`. A narrower projection breaks the coincidence, which is
    /// why this takes a position — the compiler knows the projection it asked
    /// for and emits accordingly.
    pub fn column(&self, i: usize) -> Option<&Value> {
        self.row.as_ref()?.get(i)
    }

    /// How many ordinals this cursor skipped as `Missing`/`Deleted`.
    /// Diagnostics; a nonzero count under concurrent writes is HEALTHY.
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// How many rows this cursor has materialized out of the tuple store.
    ///
    /// This is a STORAGE-READ COUNT, not a bookkeeping tally: it is incremented
    /// at the one place [`tuples::Reader::get`] returns
    /// [`Live`](RowGet::Live), so `fetched() + skipped()` is exactly the number
    /// of `get` calls the cursor has issued. That is what makes it usable as
    /// evidence that a scan is LAZY — a consumer that stops after `k` rows
    /// leaves this at `k`, while an eager implementation would show the whole
    /// collection. See `vm::exec`'s `select_is_lazy_not_materialized`.
    pub fn fetched(&self) -> u64 {
        self.fetched
    }

    /// The projection this cursor was built with — the [`ColumnId`]s it
    /// materializes, in the order [`row`](Self::row) returns them.
    ///
    /// Exposed so a caller holding a storage `ColumnId` (the VM's `Op::Column`
    /// operand) can find its POSITION rather than assume the two coincide. They
    /// do for a [`Db::scan`](crate::Db::scan) cursor and will not for a narrower
    /// projection, and that assumption would fail silently by returning the
    /// wrong column's value.
    pub fn columns(&self) -> &[ColumnId] {
        &self.columns
    }
}

#[cfg(test)]
mod tests {
    use super::Cursor;
    use crate::engine::{CollectionConfig, Db, DbOptions};
    use crate::metadata::common::{ColumnId, ColumnSpec, ColumnType, Ordinal, Row, Schema, Value};
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    const DIM: usize = 4;
    const CAPACITY: usize = 100_000;
    /// Every scalar column, in `ColumnId` order — what `Db::scan` projects.
    const ALL: [ColumnId; 3] = [0, 1, 2];
    const TEXTS: [&str; 4] = ["red", "green", "blue", "teal"];

    // ---- fixtures ---------------------------------------------------------
    //
    // Row contents are a PURE FUNCTION of the ordinal, so any thread can verify
    // any row it sees with no shared model and no lock — the same trick
    // `swmr.rs` and `tests/concurrent_stores.rs` use.

    fn vector_for(o: u64) -> Vec<f32> {
        vec![((o % 7) + 1) as f32; DIM]
    }

    /// The row's values in `ALL` order. Exactly representable, so comparisons
    /// are bit-for-bit.
    fn values_for(o: u64) -> Vec<Value> {
        vec![
            Value::Int((o % 8) as i64),
            Value::Float((o % 16) as f64 / 4.0 - 2.0),
            Value::Text(TEXTS[(o % 4) as usize].into()),
        ]
    }

    fn row_for(o: u64) -> Row {
        values_for(o)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as ColumnId, v))
            .collect()
    }

    fn schema() -> Schema {
        Schema::from_columns(vec![
            ColumnSpec::Vector {
                name: "vector".into(),
                dim: NonZeroUsize::new(DIM).unwrap(),
            },
            ColumnSpec::Scalar {
                name: "a".into(),
                ty: ColumnType::Int,
            },
            ColumnSpec::Scalar {
                name: "b".into(),
                ty: ColumnType::Float,
            },
            ColumnSpec::Scalar {
                name: "c".into(),
                ty: ColumnType::Text,
            },
        ])
        .unwrap()
    }

    fn cfgs() -> Vec<CollectionConfig> {
        vec![CollectionConfig {
            id: 0,
            name: "docs".into(),
            capacity: CAPACITY,
            schema: schema(),
        }]
    }

    /// No background flusher — checkpoints are irrelevant here and only add
    /// lock traffic.
    fn opts() -> DbOptions {
        DbOptions {
            checkpoint_interval: Duration::from_secs(3600),
        }
    }

    fn open(dir: &std::path::Path) -> Db {
        Db::open(dir, &cfgs(), opts()).unwrap()
    }

    fn insert_n(db: &Db, n: u64) {
        for o in 0..n {
            let ord = db.insert(0, &vector_for(o), row_for(o)).unwrap();
            assert_eq!(ord.0 as u64, o, "ordinals are dense from 0");
        }
    }

    /// Drain a cursor into `(ordinal, values)` pairs, verifying as it goes that
    /// every row's values are the ones its ordinal was written with.
    fn drain(cursor: &mut Cursor<'_>) -> Vec<(u64, Vec<Value>)> {
        let mut out = Vec::new();
        let mut has_row = cursor.seek_first().unwrap();
        while has_row {
            let o = cursor.ordinal().expect("parked on a row").0 as u64;
            let values = cursor.row().expect("parked on a row").to_vec();
            assert_eq!(values, values_for(o), "row {o} came back with wrong values");
            out.push((o, values));
            has_row = cursor.next().unwrap();
        }
        // Exhausted cursors report nothing, not a stale row.
        assert!(
            cursor.ordinal().is_none(),
            "exhausted cursor still has an ordinal"
        );
        assert!(cursor.row().is_none(), "exhausted cursor still has a row");
        out
    }

    fn ordinals(cursor: &mut Cursor<'_>) -> Vec<u64> {
        drain(cursor).into_iter().map(|(o, _)| o).collect()
    }

    fn bitmap_of(ordinals: impl IntoIterator<Item = u32>) -> roaring::RoaringBitmap {
        ordinals.into_iter().collect()
    }

    // ---- 1..6: deterministic behaviour ------------------------------------

    #[test]
    fn empty_collection_yields_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());

        let mut cursor = db.scan(0).unwrap();
        assert!(
            !cursor.seek_first().unwrap(),
            "empty collection must report empty"
        );
        assert!(cursor.ordinal().is_none());
        assert!(cursor.row().is_none());
        assert!(
            cursor.column(0).is_none(),
            "column is never valid before a row"
        );
        // `next` on an empty scan stays empty rather than wrapping or panicking.
        assert!(!cursor.next().unwrap());

        db.close().unwrap();
    }

    #[test]
    fn single_row_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        insert_n(&db, 1);

        let mut cursor = db.scan(0).unwrap();
        assert!(cursor.seek_first().unwrap(), "must park on the only row");
        assert_eq!(cursor.ordinal(), Some(Ordinal(0)));

        // Every field of the row, by projection position (== ColumnId here).
        let want = values_for(0);
        assert_eq!(cursor.column(0), Some(&want[0]));
        assert_eq!(cursor.column(1), Some(&want[1]));
        assert_eq!(cursor.column(2), Some(&want[2]));
        assert_eq!(cursor.column(3), None, "past the projection");
        assert_eq!(cursor.row(), Some(want.as_slice()));

        assert!(
            !cursor.next().unwrap(),
            "one row means exhausted after one advance"
        );
        assert!(cursor.ordinal().is_none());

        db.close().unwrap();
    }

    #[test]
    fn multi_row_ascending_order() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        insert_n(&db, 32);

        // Ascending order is free from the bitmap — pinned here so a later
        // refactor (a different ordinal source, a reordering optimization)
        // cannot silently take it away.
        let got = ordinals(&mut db.scan(0).unwrap());
        assert_eq!(got, (0..32).collect::<Vec<u64>>());

        db.close().unwrap();
    }

    #[test]
    fn tombstoned_rows_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        insert_n(&db, 8);
        for victim in [1u64, 4, 6] {
            db.delete(0, victim).unwrap();
        }

        // Proves the cursor is driven by `live()`, not a naive `0..len`.
        let got = ordinals(&mut db.scan(0).unwrap());
        assert_eq!(got, vec![0, 2, 3, 5, 7]);

        db.close().unwrap();
    }

    #[test]
    fn projection_returns_selected_columns() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        insert_n(&db, 4);

        // A REORDERED SUBSET: c (id 2) then a (id 0), skipping b entirely.
        let tuples = db.tuple_reader(0).unwrap();
        let live = db.metadata_reader(0).unwrap().live();
        let mut cursor = Cursor::over(live.into_iter().map(Ordinal), tuples, vec![2, 0]);

        let mut has_row = cursor.seek_first().unwrap();
        let mut seen = 0;
        while has_row {
            let o = cursor.ordinal().unwrap().0 as u64;
            let want = values_for(o);
            assert_eq!(
                cursor.row(),
                Some([want[2].clone(), want[0].clone()].as_slice()),
                "projection must be exactly the requested columns, in the requested order"
            );
            assert_eq!(cursor.column(0), Some(&want[2]));
            assert_eq!(cursor.column(1), Some(&want[0]));
            assert_eq!(cursor.column(2), None, "projection has only two columns");
            seen += 1;
            has_row = cursor.next().unwrap();
        }
        assert_eq!(seen, 4);

        db.close().unwrap();
    }

    #[test]
    fn cursor_over_arbitrary_bitmap() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        insert_n(&db, 8);

        let tuples = db.tuple_reader(0).unwrap();

        // THE SEAM. A hand-built bitmap that is NOT `live()` — every one of
        // these 8 ordinals is live, so a cursor wired to `live()` would return
        // all 8 and fail here. This is what makes WHERE/KNN reuse free.
        let bitmap = bitmap_of([3, 5]);
        let mut cursor = Cursor::over(
            bitmap.into_iter().map(Ordinal),
            tuples.clone(),
            ALL.to_vec(),
        );
        assert_eq!(ordinals(&mut cursor), vec![3, 5]);

        // …and the source is an ITERATOR, not a bitmap: a plain Vec drives the
        // same cursor. That is the shape a ranked KNN result arrives in.
        let knn = vec![Ordinal(6), Ordinal(1), Ordinal(4)];
        let mut cursor = Cursor::over(knn.into_iter(), tuples.clone(), ALL.to_vec());
        assert_eq!(
            ordinals(&mut cursor),
            vec![6, 1, 4],
            "the cursor preserves the source's order; it does not impose ascending"
        );

        // …and a BORROWED source works too — this is what `Cursor<'a>`'s
        // lifetime is for. A caller that wants to keep its bitmap (an
        // intersected WHERE mask, say) iterates it by reference; the cursor
        // then lives no longer than the bitmap. `Db::scan` hands over an owned
        // iterator instead, which is why it yields `Cursor<'static>`.
        let keep = &bitmap_of([2, 7]);
        let mut cursor = Cursor::over(keep.iter().map(Ordinal), tuples, ALL.to_vec());
        assert_eq!(ordinals(&mut cursor), vec![2, 7]);

        db.close().unwrap();
    }

    // ---- 7: the World A property under concurrency -------------------------

    /// The settled World A behaviour: with a writer running, a cursor never
    /// errors on an ordinal that transiently `get`s `Missing`/`Deleted` — it
    /// skips — and the scan still converges on the model.
    ///
    /// Structure mirrors `tests/concurrent_stores.rs`: during the concurrent
    /// phase transients are only COUNTED, never asserted on (an insert can
    /// always land between the `live()` snapshot and the `get`, so a mid-run
    /// assertion would be flaky rather than an invariant). The strong property
    /// is asserted at QUIESCENCE, once the writer has stopped and the readers
    /// have joined.
    #[test]
    fn cursor_skips_mid_apply_ordinal() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(open(dir.path()));
        let seed_rows = 64u64;
        insert_n(&db, seed_rows);

        let stop = Arc::new(AtomicBool::new(false));
        let scans = Arc::new(AtomicU64::new(0));
        let rows_read = Arc::new(AtomicU64::new(0));
        let skips = Arc::new(AtomicU64::new(0));

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(4);
        let mut handles = Vec::new();
        for role in 0..threads {
            // Role 0 runs the real full scan. The rest aim a single-ordinal
            // cursor at the FRONTIER — the ordinal the applier just published
            // into the metadata index — which is where the meta→tuple window
            // actually lives (see `concurrent_stores.rs`). A full ascending
            // scan reaches the frontier long after the window shuts, so
            // without this role the skip path would go essentially unexercised.
            let full_scan = role == 0;
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            let scans = Arc::clone(&scans);
            let rows_read = Arc::clone(&rows_read);
            let skips = Arc::clone(&skips);
            handles.push(std::thread::spawn(move || {
                let tuples = db.tuple_reader(0).expect("tuple reader");
                let meta = db.metadata_reader(0).expect("metadata reader");
                while !stop.load(Ordering::Relaxed) {
                    if full_scan {
                        let mut cursor = db.scan(0).expect("scan");
                        // `drain` verifies every row's values against its
                        // ordinal, so a torn or misfiled row fails here.
                        let rows = drain(&mut cursor).len() as u64;
                        rows_read.fetch_add(rows, Ordering::Relaxed);
                        skips.fetch_add(cursor.skipped(), Ordering::Relaxed);
                    } else {
                        let Some(frontier) = meta.live().max() else {
                            continue;
                        };
                        let mut cursor = Cursor::over(
                            std::iter::once(Ordinal(frontier)),
                            tuples.clone(),
                            ALL.to_vec(),
                        );
                        // The whole point: `seek_first` on a mid-apply ordinal
                        // returns Ok(false) — never an Err, never a panic.
                        if cursor.seek_first().expect("seek_first must not error") {
                            let o = cursor.ordinal().unwrap().0 as u64;
                            assert_eq!(cursor.row().unwrap(), values_for(o).as_slice());
                            rows_read.fetch_add(1, Ordering::Relaxed);
                        }
                        skips.fetch_add(cursor.skipped(), Ordering::Relaxed);
                    }
                    scans.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // ---- writer: randomized insert/delete on this thread ---------------
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 | 1;
        let next = |rng: &mut u64| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let mut live: Vec<u64> = (0..seed_rows).collect();
        let mut frontier = seed_rows;
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if live.is_empty() || next(&mut rng) % 100 < 55 {
                let ord = db
                    .insert(0, &vector_for(frontier), row_for(frontier))
                    .unwrap();
                assert_eq!(ord.0 as u64, frontier);
                live.push(frontier);
                frontier += 1;
            } else {
                let victim = live.swap_remove((next(&mut rng) % live.len() as u64) as usize);
                db.delete(0, victim).unwrap();
            }
        }

        stop.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().expect("reader thread panicked");
        }

        // ---- quiescence: NOW the strong property is well-defined -----------
        //
        // Every acked op was applied before its ack (`wal.rs`, commit_batch),
        // the writer is the only writer and has all its acks, and no reader is
        // running. So the stores have settled.
        let want: Vec<u64> = {
            let mut v = live.clone();
            v.sort_unstable();
            v
        };
        let got = ordinals(&mut db.scan(0).unwrap());
        assert_eq!(
            got, want,
            "settled scan must be exactly the writer's live set"
        );

        // …and nothing in the allocated range is left Missing: every ordinal is
        // either a live row with the right values or a tombstone.
        let tuples = db.tuple_reader(0).unwrap();
        let live_set: std::collections::BTreeSet<u64> = want.iter().copied().collect();
        for o in 0..frontier {
            match tuples.get(Ordinal(o as u32), &ALL).unwrap() {
                crate::metadata::tuples::RowGet::Live(values) => {
                    assert_eq!(values, values_for(o));
                    assert!(live_set.contains(&o), "ordinal {o} live but not scanned");
                }
                crate::metadata::tuples::RowGet::Deleted => {
                    assert!(!live_set.contains(&o), "ordinal {o} scanned but tombstoned");
                }
                crate::metadata::tuples::RowGet::Missing => {
                    panic!("ordinal {o} still Missing after the run quiesced")
                }
            }
        }

        // A run where the readers never got going would prove nothing.
        assert!(scans.load(Ordering::Relaxed) > 1_000, "readers barely ran");
        assert!(
            rows_read.load(Ordering::Relaxed) > 1_000,
            "readers saw almost no rows"
        );
        assert!(frontier > seed_rows, "writer made no progress");

        // Skips are REPORTED, not asserted: `tests/concurrent_stores.rs` owns
        // the "the race is observable" assertion, and duplicating it here would
        // only add a host-sensitive failure mode. The deterministic skip path is
        // covered by `tombstoned_rows_skipped`.
        eprintln!(
            "[cursor] {} scans, {} rows, {} skipped transients",
            scans.load(Ordering::Relaxed),
            rows_read.load(Ordering::Relaxed),
            skips.load(Ordering::Relaxed),
        );

        Arc::into_inner(db)
            .expect("readers joined")
            .close()
            .unwrap();
    }
}
