# Flats for C

The C ABI every other binding is built on: one header,
[`include/flats.h`](include/flats.h), and one library. Use it from C or C++,
or from any language that can call C.

For V-SQL and the concepts shared by every SDK, see the
[user guide](../../docs/guide.md).

## Setup

```sh
cargo build --release -p flats-c
# → target/release/libflats_c.so  (shared)
#   target/release/libflats_c.a   (static)

cc app.c -I sdk/c/include -L target/release -lflats_c \
   -Wl,-rpath,$PWD/target/release -o app
```

Linking the static `libflats_c.a` instead also needs
`-lpthread -ldl -lm`.

## Quickstart

```c
#include <stdio.h>
#include <string.h>
#include "flats.h"

static int fail(const char *what) {
    fprintf(stderr, "%s: %s\n", what, flats_last_error());
    return 1;
}

int main(void) {
    flats_db *db;
    if (flats_open("./my-db", &db) != FLATS_OK) return fail("open");

    if (flats_execute(db,
            "CREATE COLLECTION articles (id INT PRIMARY KEY, vector VECTOR(4), title TEXT) "
            "WITH (capacity = 10000);", NULL) != FLATS_OK)
        return fail("create");

    /* Load rows: vectors go in as binary floats, no SQL text. */
    flats_batch *b;
    flats_batch_new(db, "articles", &b);
    const char *titles[] = {"alpha", "beta", "gamma"};
    for (int i = 0; i < 3; i++) {
        float v[4] = {1.0f - i * 0.4f, i * 0.4f, 0.0f, 0.0f};
        flats_batch_vector(b, v, 4);
        flats_batch_int(b, "id", i);
        flats_batch_text(b, "title", titles[i], strlen(titles[i]));
        if (flats_batch_end_row(b) != FLATS_OK) return fail("row");
    }
    size_t inserted;
    if (flats_batch_commit(b, &inserted) != FLATS_OK) return fail("commit");
    flats_batch_free(b);

    /* Search, and read the result. */
    flats_result *res;
    if (flats_execute(db,
            "SEARCH TOP 2 NEAREST TO [1, 0, 0, 0] FROM articles RETURNING id, title, score;",
            &res) != FLATS_OK)
        return fail("search");
    for (size_t r = 0; r < flats_result_row_count(res); r++) {
        size_t len;
        const char *title = flats_result_text(res, r, 1, &len);
        printf("%lld %s %.3f\n", (long long)flats_result_int(res, r, 0), title,
               flats_result_float(res, r, 2));
    }
    flats_result_free(res);

    return flats_close(db) == FLATS_OK ? 0 : fail("close");
}
```

Output:

```
0 alpha 1.000
1 beta 0.600
```

## How the API works

**Status codes.** Everything that can fail returns an `int`: `FLATS_OK` (0),
or one of

| Code | Means |
|---|---|
| `FLATS_ERR_INVALID_ARGUMENT` | NULL handle or string, invalid UTF-8, a value the schema refuses (wrong vector length, wrong type, NaN, unknown column) |
| `FLATS_ERR_QUERY` | Bad or unsupported V-SQL, unknown collection or column |
| `FLATS_ERR_ENGINE` | Duplicate primary key, collection full, backup target not empty, I/O |
| `FLATS_ERR_INTERNAL` | A bug in Flats |

**Error messages.** After a failure, `flats_last_error()` describes it. The
message is per thread and stays valid until that thread's next failing call,
so copy it if you need it later.

**Handles.** Three opaque types, each with its own release function:

| Handle | Made by | Released by |
|---|---|---|
| `flats_db *` | `flats_open` | `flats_close` (checkpoints) |
| `flats_result *` | `flats_execute` | `flats_result_free` |
| `flats_batch *` | `flats_batch_new` | `flats_batch_free` |

Releasing `NULL` is a no-op. `flats_close` waits for any calls still running
on that database. Results and cursors don't depend on the database once you
have them. A batch whose database has been closed refuses to commit with
`FLATS_ERR_INVALID_ARGUMENT` instead of crashing.

**Strings** you pass in are NUL-terminated UTF-8, except the value passed to
`flats_batch_text`, which takes an explicit length. Strings you get back
(column names, text cells) are NUL-terminated and owned by the result.

**Threads.** A `flats_db *` can be used from many threads at once; searches
run in parallel. A result or a batch must be used by one thread at a time.

## Running statements

`flats_execute(db, sql, &res)` runs one V-SQL statement. Reads (`SELECT`,
`SEARCH`) fill the result with rows; writes produce an empty one. Pass `NULL`
for the result if you don't want it.

A write is durable when `flats_execute` returns `FLATS_OK`.

## Parameters

Values from outside your program go in as parameters, never pasted into the
SQL text. Write `?` for each value and pass a `flats_param` array:

```c
flats_param p[3];
memset(p, 0, sizeof p);
p[0].type = FLATS_TYPE_VECTOR; p[0].vector = query;  p[0].vector_len = 768;
p[1].type = FLATS_TYPE_TEXT;   p[1].text = lang;     p[1].text_len = strlen(lang);
p[2].type = FLATS_TYPE_INT;    p[2].int_value = 2023;

flats_execute_params(db,
    "SEARCH TOP 10 NEAREST TO ? FROM docs WHERE lang = ? AND year >= ? RETURNING id, score;",
    p, 3, &res);
```

A parameter is a value, never statement text, so a hostile string can't
change what the statement does. Set `type` plus the matching field.
`text` takes an explicit length and need not be NUL-terminated. A vector
parameter is passed as raw floats, so it's exact and fast. The `?` count must
match `n` (`FLATS_ERR_QUERY` otherwise). The semantics are in the
[guide](../../docs/guide.md#parameters-how-values-get-in).

## Reading results

```c
size_t cols = flats_result_column_count(res);
size_t rows = flats_result_row_count(res);
const char *name = flats_result_column_name(res, c);

switch (flats_result_type(res, r, c)) {
case FLATS_TYPE_INT:    flats_result_int(res, r, c);              break;
case FLATS_TYPE_FLOAT:  flats_result_float(res, r, c);            break;
case FLATS_TYPE_TEXT:   flats_result_text(res, r, c, &len);       break;  /* UTF-8 */
case FLATS_TYPE_VECTOR: flats_result_vector(res, r, c, &len);     break;  /* float[len] */
}
```

The accessors never crash on a bad index: out of range, or the wrong type,
gives `0` or `NULL`. Everything they return lives until `flats_result_free`.

## Streaming large results

`flats_execute` collects every row into one result. For a large read, use a
cursor: `flats_query` runs the statement, and each `flats_cursor_next` returns
the next chunk as an ordinary `flats_result`. A chunk with zero rows means
the cursor is done:

```c
flats_cursor *cur;
flats_query(db, "SELECT id, title FROM docs WHERE year > ?;", p, 1, &cur);
for (;;) {
    flats_result *chunk;
    if (flats_cursor_next(cur, 1000, &chunk) != FLATS_OK) break;
    size_t n = flats_result_row_count(chunk);
    /* ... read rows 0..n with the usual accessors ... */
    flats_result_free(chunk);
    if (n == 0) break;
}
flats_cursor_free(cur);
```

Rows are read as they're pulled, so memory is bounded by the chunk size. A
cursor reads one snapshot and, like a result, doesn't depend on the
`flats_db` once you have it. For paging, use `LIMIT n OFFSET m` instead.

## Loading data with batches

The batch API is the fast way in. It skips SQL parsing entirely and commits
every finished row as a single statement:

1. `flats_batch_new(db, "collection", &b)`
2. per row: `flats_batch_vector`, then one `flats_batch_int` /
   `_float` / `_text` per column (in any order), then
   `flats_batch_end_row`
3. `flats_batch_commit(b, &n)`, as often as you like (every 10,000 rows is a
   good rhythm), and finally `flats_batch_free(b)`.

Every value is checked against the schema as you add it, so a wrong type or
length fails on the call that supplied it. `flats_batch_end_row` fails if a
column is missing, and drops that half-built row. `flats_batch_commit` is
all-or-nothing: a duplicate key anywhere refuses every row in the batch. The
batch is emptied either way and can be reused.

`flats_batch_column_type(b, "col")` reports a column's type. It's useful when
your values come untyped, from JSON or a scripting language.

## Backups

```c
if (flats_backup(db, "/backups/my-db") != FLATS_OK) fail("backup");
```

This writes a consistent copy while the database stays open. Open the copy to
restore it. The destination must not exist or must be empty.

## Tests

`sdk/c/tests/smoke.c` is a complete program covering every part of the API.
`cargo test -p flats-c` compiles it against the header with your C compiler
and runs it, alongside Rust tests of every function.
