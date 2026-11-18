use thiserror::Error;

use crate::metadata::common::ColumnType;

/// Every failure the storage engine reports: I/O and durability, file-format
/// checks, and the refusals that keep a collection consistent (dimension,
/// capacity, schema, primary key).
///
/// The query pipeline reports its own stage-local errors through
/// [`QueryError`](crate::QueryError); an engine failure reached through
/// `Db::execute` arrives wrapped in it.
#[derive(Error, Debug)]
pub enum Error {
    /// A vector's length is not the collection's dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The collection's dimension.
        expected: usize,
        /// The length supplied.
        got: usize,
    },

    /// A vector column was declared with dimension 0.
    #[error("dimension cannot be 0")]
    InvalidDimension,

    /// A search asked for zero results.
    #[error("invalid top k: {k} - k should be a natural number > 0")]
    InvalidTopK {
        /// The `k` requested.
        k: usize,
    },

    /// An operating-system I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// An index file's header failed its integrity check.
    #[error("index file's header is corrupt")]
    CorruptHeader,

    /// A file is not a Flats file (its magic number is wrong).
    #[error("bad magic number: {got}")]
    BadMagic {
        /// The magic number found.
        got: u32,
    },

    /// A file was written by a format version this build does not read.
    #[error("unsupported index file version: {got}")]
    UnsupportedVersion {
        /// The version found.
        got: u32,
    },

    /// The collection has no room for the rows being inserted. Capacity is
    /// fixed at creation; compaction reclaims the slots of deleted rows.
    #[error("collection is at capacity: {capacity}")]
    CapacityExceeded {
        /// The collection's capacity.
        capacity: usize,
    },

    /// `dim × capacity` does not fit in memory addressing.
    #[error("capacity overflow: dim={dim}, capacity={capacity} exceeds usize")]
    CapacityOverflow {
        /// The dimension requested.
        dim: usize,
        /// The capacity requested.
        capacity: usize,
    },

    /// No collection has this id.
    #[error("unknown collection: {id}")]
    UnknownCollection {
        /// The id looked up.
        id: u32,
    },

    /// A scan enumerated an ordinal from a liveness snapshot and the tuple
    /// store had no row for it.
    ///
    /// This is a CONSISTENCY FAILURE, not a transient. A snapshot only ever
    /// admits rows the applier finished writing — it publishes liveness last,
    /// after the vector and the values are both in place — so an ordinal that
    /// is in a snapshot and missing from the tuple store means those two
    /// disagree. It used to be legal and routine, back when liveness had three
    /// owners updated at three different moments; it is now a bug in the
    /// applier's ordering, and silently skipping it would hide exactly the
    /// class of defect this became an error to catch.
    #[error(
        "row {ordinal} of collection {collection} is in the liveness snapshot but absent from the tuple store"
    )]
    SnapshotRowMissing {
        /// The collection scanned.
        collection: u32,
        /// The ordinal the tuple store lacked.
        ordinal: u32,
    },

    /// The database is halted: a record was fsynced, then failed to apply to
    /// the indexes. It is committed and will replay on the next `Db::open`, but
    /// the index does not reflect it and a watermark cannot express a gap — so
    /// every further write is refused, and so is every checkpoint and
    /// truncation (either would shed the frame recovery needs).
    ///
    /// Reads keep working against the last consistent prefix of the log. The
    /// remedy is to reopen the database, which replays the tail; apply is
    /// idempotent, so replay is always safe. `lsn` names the unapplied record
    /// and `cause` is the original apply failure.
    #[error("database halted at lsn {lsn}: {cause} (reopen to replay the unapplied record)")]
    Poisoned {
        /// The log position of the record that failed to apply.
        lsn: u64,
        /// The original apply failure.
        cause: String,
    },

    /// One statement's installed payload exceeds the per-record cap.
    ///
    /// A `Record::Mutate` is deliberately ONE frame with ONE checksum behind
    /// ONE fsync — that is what makes a multi-row DELETE or UPDATE all-or-
    /// nothing. The cost of that guarantee is that the whole statement must fit
    /// in memory: on the commit thread now, and on every replay for the life of
    /// the log. The cap turns "this UPDATE rewrites a million vectors" into a
    /// named refusal at the door rather than an allocation failure inside the
    /// commit loop, and it bounds how long the metadata lock is held, since
    /// that is proportional to statement size.
    ///
    /// Split the statement with a narrower predicate.
    #[error("statement installs {bytes} bytes, over the {cap}-byte per-record cap")]
    StatementTooLarge {
        /// The statement's installed payload.
        bytes: usize,
        /// The cap it exceeded.
        cap: usize,
    },

    /// A collection NAME did not resolve to a registered collection. Distinct
    /// from `UnknownCollection`, which reports an id: here there is no id to
    /// report — failing to produce one is the error. Raised by
    /// `Db::collection_id`, the bridge from the name bytecode carries to the id
    /// every other engine method is keyed by.
    #[error("unknown collection: {name}")]
    UnknownCollectionName {
        /// The name looked up.
        name: String,
    },

    /// A config supplied to `Db::open` disagrees (dim, capacity, or schema)
    /// with what the persisted catalog has registered under the same id.
    #[error("collection {id}'s config conflicts with the persisted catalog")]
    CollectionConfigMismatch {
        /// The conflicting collection's id.
        id: u32,
    },

    /// A collection with this name already exists.
    #[error("collection name already exists: {name}")]
    CollectionExists {
        /// The name requested.
        name: String,
    },

    /// A collection name was empty.
    #[error("collection name must be non-empty")]
    InvalidCollectionName,

    /// A collection was declared with capacity 0.
    #[error("capacity must be > 0")]
    InvalidCapacity,

    // -- metadata layer (Phase 4) ------------------------------------------
    // Callers in Phase 4c only distinguish "retryable via WAL replay" (Io)
    // from "caller bug" (everything below). Keep variants coarse.
    /// A row named a column the schema does not have.
    #[error("unknown column: {column}")]
    UnknownColumn {
        /// The column id.
        column: u32,
    },

    /// A value's type is not its column's.
    #[error("type mismatch on column {column}: expected {expected:?}, got {got:?}")]
    TypeMismatch {
        /// The column id.
        column: u32,
        /// The column's declared type.
        expected: ColumnType,
        /// The value's type.
        got: ColumnType,
    },

    /// NaN handed to a Float column. Rejected at the boundary so the
    /// BTreeMap ordering invariant can never be violated.
    #[error("NaN rejected for float column {column}")]
    NaNRejected {
        /// The column id.
        column: u32,
    },

    /// Row missing a column, or has extras/duplicates (no NULLs in Phase 4).
    #[error("row does not match schema (every column exactly once, no NULLs)")]
    IncompleteRow,

    /// A schema declared two columns with the same name.
    #[error("duplicate column in schema: {0}")]
    DuplicateColumn(String),

    /// A schema must declare exactly one VECTOR column — every collection is
    /// provisioned with exactly one flat vector index.
    #[error("schema must have exactly one vector column, found {found}")]
    VectorColumnCount {
        /// How many vector columns were declared.
        found: usize,
    },

    /// A schema declared more than one PRIMARY KEY. V-SQL has no composite
    /// keys: one column names the row, or none does.
    #[error("schema must declare at most one primary key, found {found}")]
    PrimaryKeyCount {
        /// How many primary keys were declared.
        found: usize,
    },

    /// A column that cannot serve as an identity was declared PRIMARY KEY.
    #[error("column {column:?} of type {ty} cannot be a primary key (TEXT or INT only)")]
    PrimaryKeyType {
        /// The offending column.
        column: String,
        /// Its declared type.
        ty: String,
    },

    /// A statement carrying a replacement embedding matched more than one row.
    ///
    /// The binder requires equality on the primary key for exactly this
    /// reason, and the applier keeps that key unique — so reaching this means
    /// those two guarantees disagree, not that the caller wrote a loose
    /// predicate.
    #[error("a replacement embedding matched {matched} rows; it may name only one")]
    KeyMatchedManyRows {
        /// How many rows the predicate matched.
        matched: u64,
    },

    /// An insert or install carries a primary key some LIVE row already holds.
    #[error("a live row already holds primary key {value} for column {column:?}")]
    DuplicateKey {
        /// The key column's name.
        column: String,
        /// The offending value, rendered for the message.
        value: String,
    },

    /// Snapshot file exists but failed magic/version/CRC checks.
    /// Phase 4a/4b policy: `open()` swallows this internally ("start empty at
    /// lsn 0, WAL replay rebuilds") — the variant exists for the cases that
    /// must NOT be swallowed, e.g. a schema that contradicts the caller's.
    #[error("corrupt snapshot: {0}")]
    CorruptSnapshot(String),

    /// The database directory is already open — by another process, or by
    /// another handle in this one. One handle per directory: two would each
    /// act as the single writer and corrupt the files between them.
    #[error("database {path} is locked: another handle already has it open")]
    DatabaseLocked {
        /// The lock file that could not be taken.
        path: String,
    },

    /// A persisted snapshot's schema contradicts the one the caller supplied.
    #[error("snapshot schema does not match the schema supplied by the caller")]
    SchemaMismatch,
}

/// `Result` with the engine's [`Error`](enum@Error).
pub type Result<T> = std::result::Result<T, Error>;
