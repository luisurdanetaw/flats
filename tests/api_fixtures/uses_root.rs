//! A downstream crate that uses ONLY the crate-root API. It must compile
//! against `flats` with default features — if it stops compiling, a type a
//! real caller needs has fallen out of the public surface.

use flats::{
    ColumnId, ColumnSpec, ColumnType, CollectionConfig, CollectionId, CompactionOptions, Cursor,
    Db, DbOptions, Error, Ordinal, OutputRow, QueryError, RangeOp, RegValue, Result, Row,
    RowStream, Schema, SearchResult, Value,
};

fn workflow(dir: &std::path::Path) -> Result<()> {
    let opts = DbOptions {
        compaction: CompactionOptions::default(),
        ..DbOptions::default()
    };
    let db = Db::open(dir, &[], opts)?;
    let schema = Schema::from_columns(vec![
        ColumnSpec::Scalar { name: "id".into(), ty: ColumnType::Int },
        ColumnSpec::Vector { name: "vector".into(), dim: std::num::NonZeroUsize::new(2).unwrap() },
    ])?;
    let id: CollectionId = db.create_collection("raw", 8, schema)?;
    let key: ColumnId = 0;
    let row: Row = vec![(key, Value::Int(1))];
    let one: Ordinal = db.insert(id, &[1.0, 0.0], row)?;
    let many: Vec<Ordinal> = db.insert_many(id, vec![(vec![0.0, 1.0], vec![(key, Value::Int(2))])])?;
    let hits: Vec<SearchResult> = db.search(id, &[1.0, 0.0], 1)?;
    let _ = (one, many, hits[0].id, hits[0].score, RangeOp::Lt);
    let cursor: Cursor<'static> = db.scan(id)?;
    drop(cursor);
    let configs: Vec<CollectionConfig> = db.collections();
    let _ = configs[0].name.clone();

    let stream: std::result::Result<RowStream, QueryError> = db.execute("SELECT id FROM raw;");
    for row in stream.map_err(|e| Error::Io(std::io::Error::other(e.to_string())))? {
        let row: OutputRow = row.map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
        if let [RegValue::Int(n)] = row.0.as_slice() {
            let _ = n;
        }
    }
    let _ = flats::dot(&[1.0], &[1.0]);
    db.backup(dir.join("backup"))?;
    db.drop_collection("raw")?;
    db.checkpoint()?;
    db.close()
}

fn main() {
    let _ = workflow(std::path::Path::new("unused"));
}
