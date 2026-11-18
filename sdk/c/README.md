# Flats — C ABI

`include/flats.h` is the contract; `libflats_c.so` (or the static
`libflats_c.a`) implements it. Every other binding wraps these functions.

```sh
cargo build --release -p flats-c        # → target/release/libflats_c.{so,a}
cc app.c -I sdk/c/include -L target/release -lflats_c
```

```c
flats_db *db;
if (flats_open("./my-db", &db) != FLATS_OK) {
    fprintf(stderr, "%s\n", flats_last_error());
}
flats_execute(db, "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(3)) "
                  "WITH (capacity = 1000);", NULL);

/* Load rows without SQL text: vectors go in as binary. */
flats_batch *b;
flats_batch_new(db, "docs", &b);
float v[3] = {0.1f, 0.2f, 0.3f};
flats_batch_vector(b, v, 3);
flats_batch_int(b, "id", 1);
flats_batch_end_row(b);
flats_batch_commit(b, NULL);           /* one statement: all rows or none */
flats_batch_free(b);

flats_result *res;
flats_execute(db, "SEARCH TOP 5 NEAREST TO [0.1, 0.2, 0.3] FROM docs RETURNING id, score;", &res);
for (size_t r = 0; r < flats_result_row_count(res); r++)
    printf("%lld %f\n", (long long)flats_result_int(res, r, 0), flats_result_float(res, r, 1));
flats_result_free(res);
flats_close(db);
```

Conventions (status codes, the per-thread last error, handle ownership,
thread safety) are spelled out at the top of the header. `tests/smoke.c` is a
complete program, compiled and run by `cargo test -p flats-c`.
