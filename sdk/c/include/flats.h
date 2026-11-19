/*
 * flats.h — the C ABI for Flats, a tiny embeddable vector database.
 *
 * Every language binding (Python via ctypes, Node.js via N-API) wraps exactly
 * these functions, so this file is the one contract they share.
 *
 * CONVENTIONS
 *
 *   Status.   Functions that can fail return an int: FLATS_OK (0) on success,
 *             or a nonzero FLATS_ERR_* code. On failure, flats_last_error()
 *             describes what went wrong.
 *
 *   Errors.   flats_last_error() is PER THREAD and describes the most recent
 *             failed call made on that thread. The pointer stays valid until
 *             the next flats_* call on the same thread. Copy it if you need it
 *             longer. It is NULL when that thread has seen no failure.
 *
 *   Strings.  Every `const char *` argument is NUL-terminated UTF-8.
 *             Invalid UTF-8 is FLATS_ERR_INVALID_ARGUMENT.
 *
 *   Handles.  Opaque. A flats_db may be shared across threads: every function
 *             taking one is safe to call concurrently. A flats_result and a
 *             flats_batch belong to one thread at a time. Each handle is
 *             released by its own *_free / flats_close; passing NULL to those
 *             is a no-op. Using a handle after releasing it is undefined.
 *             flats_close waits for calls already running on that database
 *             to finish. Results and cursors don't depend on the database
 *             once you have them. A batch whose database has been closed
 *             refuses commits with FLATS_ERR_INVALID_ARGUMENT rather than
 *             crashing; free it as usual.
 *
 *   Memory.   Pointers returned INTO a result (column names, text, vectors)
 *             are owned by that result and live until flats_result_free.
 *
 *   Panics.   No Rust panic ever crosses this boundary; one that would is
 *             reported as FLATS_ERR_INTERNAL.
 */
#ifndef FLATS_H
#define FLATS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes ------------------------------------------------------ */

#define FLATS_OK                   0
/* A NULL where a handle or string was required, invalid UTF-8, or a
 * row/column index out of range. */
#define FLATS_ERR_INVALID_ARGUMENT 1
/* The statement did not parse, bind, or compile; or it named something that
 * does not exist. Nothing was changed. */
#define FLATS_ERR_QUERY            2
/* The engine refused or failed the operation: a duplicate primary key, a
 * full collection, a wrong dimension, an I/O failure. */
#define FLATS_ERR_ENGINE           3
/* A bug in Flats (a caught panic). */
#define FLATS_ERR_INTERNAL         4
/* The statement was stopped before it finished: flats_interrupt, or its
 * timeout. Nothing it would have written was written. */
#define FLATS_ERR_INTERRUPTED      5

/* ---- cell types (flats_result_type) ------------------------------------ */

#define FLATS_TYPE_INT    1  /* int64_t           */
#define FLATS_TYPE_FLOAT  2  /* double            */
#define FLATS_TYPE_TEXT   3  /* UTF-8 bytes       */
#define FLATS_TYPE_VECTOR 4  /* array of float    */

typedef struct flats_db     flats_db;
typedef struct flats_result flats_result;
typedef struct flats_batch  flats_batch;
typedef struct flats_cursor flats_cursor;

/* ---- library ----------------------------------------------------------- */

/* The library version, e.g. "0.1.0". Static; never free it. */
const char *flats_version(void);

/* The calling thread's most recent error message, or NULL. See CONVENTIONS. */
const char *flats_last_error(void);

/* ---- database ---------------------------------------------------------- */

/* Open (creating if missing) the database in directory `path`. Runs crash
 * recovery before returning. */
int flats_open(const char *path, flats_db **out);

/* flats_open for a database that refuses to open because a collection's
 * store file is damaged or missing (FLATS_ERR_ENGINE, "corrupt snapshot:
 * ... salvage"). Each damaged collection opens with only the rows written
 * since its last checkpoint; the damaged file is kept as <name>.damaged.
 * Restore a backup instead if you have one. On a healthy database it is the
 * same as flats_open. */
int flats_open_salvage(const char *path, flats_db **out);

/* Checkpoint and close. Returns the checkpoint's status; the handle is
 * released either way. NULL is a no-op returning FLATS_OK. */
int flats_close(flats_db *db);

/* Run ONE V-SQL statement. For a read (SELECT, SEARCH) the rows are
 * collected into *out; for a write *out holds an empty result. Pass
 * out = NULL to discard the result. A write is durable when this returns
 * FLATS_OK. */
int flats_execute(flats_db *db, const char *sql, flats_result **out);

/* ---- parameters --------------------------------------------------------
 *
 * The SAFE way to put values into a statement. Write `?` wherever a literal
 * would go, and pass the values here, in order. A parameter is a VALUE, never
 * statement text: nothing inside a string parameter — quotes, `;`, keywords —
 * can change what the statement does. Parameters stand only where literals
 * can, never for a collection or column name. Never build V-SQL by pasting
 * untrusted strings into it; use parameters.
 *
 *   flats_param p[2] = {0};
 *   p[0].type = FLATS_TYPE_VECTOR; p[0].vector = emb;  p[0].vector_len = 768;
 *   p[1].type = FLATS_TYPE_TEXT;   p[1].text = lang;   p[1].text_len = strlen(lang);
 *   flats_execute_params(db,
 *       "SEARCH TOP 10 NEAREST TO ? FROM docs WHERE lang = ? RETURNING id, score;",
 *       p, 2, &res);
 *
 * Set `type` and the field(s) for it; the others are ignored. Text need not be
 * NUL-terminated. A vector parameter is exact — no float formatting — and is
 * the fast way to pass a query embedding.
 */
typedef struct flats_param {
    int type;            /* FLATS_TYPE_INT / _FLOAT / _TEXT / _VECTOR */
    int64_t int_value;   /* INT                                        */
    double float_value;  /* FLOAT (must be finite)                     */
    const char *text;    /* TEXT: UTF-8 bytes ...                      */
    size_t text_len;     /*       ... and their length                 */
    const float *vector; /* VECTOR: components ...                     */
    size_t vector_len;   /*         ... and their count                */
} flats_param;

/* flats_execute with `?` placeholders filled from params[0..n]. The count
 * must match the placeholders (FLATS_ERR_QUERY otherwise). params may be NULL
 * when n is 0. */
int flats_execute_params(flats_db *db, const char *sql, const flats_param *params, size_t n,
                         flats_result **out);

/* ---- cursors -------------------------------------------------------------
 *
 * flats_execute collects every row into one result. For a large read, stream
 * it instead: flats_query runs the statement and returns a cursor, and each
 * flats_cursor_next hands back the next chunk of at most max_rows rows as an
 * ordinary flats_result (same accessors, same column names in every chunk). A
 * chunk with 0 rows means the cursor is exhausted. Rows are read from storage
 * as they are pulled, so memory stays bounded by the chunk size.
 *
 *   flats_cursor *cur;
 *   flats_query(db, "SELECT id, title FROM docs WHERE year > ?;", p, 1, &cur);
 *   for (;;) {
 *       flats_result *chunk;
 *       if (flats_cursor_next(cur, 1000, &chunk) != FLATS_OK) break;
 *       size_t n = flats_result_row_count(chunk);
 *       ...                                  // read rows 0..n
 *       flats_result_free(chunk);
 *       if (n == 0) break;
 *   }
 *   flats_cursor_free(cur);
 *
 * The cursor reads ONE snapshot: rows written after flats_query returns are
 * not in it. Like a result, it does not depend on its flats_db once made. A
 * write statement runs at flats_query and yields an exhausted cursor.
 */
int flats_query(flats_db *db, const char *sql, const flats_param *params, size_t n,
                flats_cursor **out);
/* The next chunk: at most max_rows (>= 1) rows; 0 rows when exhausted. */
int flats_cursor_next(flats_cursor *cur, size_t max_rows, flats_result **out);
void flats_cursor_free(flats_cursor *cur);

/* Write a consistent copy of the database into directory `dest`, which must
 * not exist or must be empty. Restore by opening `dest`. */
int flats_backup(flats_db *db, const char *dest);

/* ---- limits and cancellation -------------------------------------------- */

/* Stop every statement running on db right now (searches, scans, cursors)
 * with FLATS_ERR_INTERRUPTED at its next check — within a fraction of a
 * millisecond for a search. Statements started afterwards are unaffected.
 * Safe to call from any thread, e.g. a watchdog. */
int flats_interrupt(flats_db *db);

/* Stop any statement that runs longer than ms milliseconds with
 * FLATS_ERR_INTERRUPTED. Negative disables (the default). Applies to
 * statements started after the call. */
int flats_set_statement_timeout_ms(flats_db *db, int64_t ms);

/* Hard caps, independent of any setting: a WHERE may nest parentheses at most
 * 64 deep and hold at most 4096 comparisons (FLATS_ERR_QUERY beyond); a
 * collection may declare at most 65536 dimensions and reserve at most 4 TiB of
 * vector storage (FLATS_ERR_ENGINE beyond, before anything is written). */

/* ---- results ----------------------------------------------------------- */

size_t flats_result_column_count(const flats_result *res);
/* NULL if col is out of range. */
const char *flats_result_column_name(const flats_result *res, size_t col);
size_t flats_result_row_count(const flats_result *res);
/* One of FLATS_TYPE_*, or 0 if (row, col) is out of range. */
int flats_result_type(const flats_result *res, size_t row, size_t col);
/* The cell's value. 0 if out of range or not of that type. */
int64_t flats_result_int(const flats_result *res, size_t row, size_t col);
double flats_result_float(const flats_result *res, size_t row, size_t col);
/* A TEXT cell: NUL-terminated UTF-8, its byte length (excluding the NUL) in
 * *len when len is not NULL. NULL if out of range or not TEXT. */
const char *flats_result_text(const flats_result *res, size_t row, size_t col, size_t *len);
/* A VECTOR cell: its components, the count in *len when len is not NULL.
 * NULL if out of range or not VECTOR. */
const float *flats_result_vector(const flats_result *res, size_t row, size_t col, size_t *len);
void flats_result_free(flats_result *res);

/* ---- batch insert ------------------------------------------------------ */
/*
 * The fast path for loading rows: no V-SQL text, vectors passed as binary.
 * Build rows one at a time, then commit them all as ONE statement: every row
 * lands, or none does (a duplicate key anywhere refuses the whole batch).
 *
 *   flats_batch *b;
 *   flats_batch_new(db, "docs", &b);
 *   for (...) {
 *       flats_batch_vector(b, emb, 768);
 *       flats_batch_text(b, "id", "d1", 2);
 *       flats_batch_int(b, "year", 2024);
 *       flats_batch_end_row(b);
 *   }
 *   flats_batch_commit(b, &n);    // b is empty again and reusable
 *   flats_batch_free(b);
 *
 * Every column, the primary key included, must be set once per row. Values
 * are checked against the collection's schema as they are added.
 */
int flats_batch_new(flats_db *db, const char *collection, flats_batch **out);
int flats_batch_vector(flats_batch *b, const float *data, size_t len);
int flats_batch_int(flats_batch *b, const char *column, int64_t value);
int flats_batch_float(flats_batch *b, const char *column, double value);
int flats_batch_text(flats_batch *b, const char *column, const char *utf8, size_t len);
/* The type a column takes — FLATS_TYPE_* (VECTOR for the vector column) —
 * or 0 if the collection has no such column. For bindings whose language has
 * one number type and must pick flats_batch_int or flats_batch_float. */
int flats_batch_column_type(const flats_batch *b, const char *column);
/* Finish the current row. Refused if any column is unset. */
int flats_batch_end_row(flats_batch *b);
/* Rows finished and not yet committed. */
size_t flats_batch_len(const flats_batch *b);
/* Insert every finished row as one statement. *inserted (if not NULL)
 * receives the count. The batch is emptied on success AND on failure. */
int flats_batch_commit(flats_batch *b, size_t *inserted);
void flats_batch_free(flats_batch *b);

#ifdef __cplusplus
}
#endif

#endif /* FLATS_H */
