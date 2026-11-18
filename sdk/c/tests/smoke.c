/* A C program using flats.h end to end. Built and run by tests/c_smoke.rs;
 * prints one line per check and exits nonzero on the first failure. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flats.h"

#define CHECK(cond, what)                                                  \
    do {                                                                   \
        if (!(cond)) {                                                     \
            const char *e = flats_last_error();                            \
            fprintf(stderr, "FAIL %s: %s\n", what, e ? e : "(no error)");  \
            return 1;                                                      \
        }                                                                  \
        printf("ok %s\n", what);                                           \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: smoke DB_DIR BACKUP_DIR\n");
        return 2;
    }
    flats_db *db = NULL;
    CHECK(flats_open(argv[1], &db) == FLATS_OK && db, "open");
    CHECK(strcmp(flats_version(), "0.1.0") == 0, "version");

    CHECK(flats_execute(db,
              "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(4), title TEXT) "
              "WITH (capacity = 100);", NULL) == FLATS_OK, "create");

    flats_batch *b = NULL;
    CHECK(flats_batch_new(db, "docs", &b) == FLATS_OK, "batch_new");
    for (int i = 0; i < 10; i++) {
        float v[4] = {(float)i, 1.0f, 0.0f, 0.5f};
        char title[16];
        snprintf(title, sizeof title, "doc %d", i);
        CHECK(flats_batch_vector(b, v, 4) == FLATS_OK, "batch_vector");
        CHECK(flats_batch_int(b, "id", i) == FLATS_OK, "batch_int");
        CHECK(flats_batch_text(b, "title", title, strlen(title)) == FLATS_OK, "batch_text");
        CHECK(flats_batch_end_row(b) == FLATS_OK, "batch_end_row");
    }
    size_t n = 0;
    CHECK(flats_batch_commit(b, &n) == FLATS_OK && n == 10, "batch_commit");
    flats_batch_free(b);

    flats_result *res = NULL;
    CHECK(flats_execute(db,
              "SEARCH TOP 3 NEAREST TO [9, 1, 0, 0.5] FROM docs RETURNING id, title, vector;",
              &res) == FLATS_OK, "search");
    CHECK(flats_result_row_count(res) == 3, "three hits");
    CHECK(flats_result_column_count(res) == 3, "three columns");
    CHECK(strcmp(flats_result_column_name(res, 1), "title") == 0, "column name");
    CHECK(flats_result_type(res, 0, 0) == FLATS_TYPE_INT && flats_result_int(res, 0, 0) == 9,
          "nearest is id 9");
    size_t len = 0;
    const char *title = flats_result_text(res, 0, 1, &len);
    CHECK(title && len == 5 && strcmp(title, "doc 9") == 0, "text cell");
    const float *vec = flats_result_vector(res, 0, 2, &len);
    CHECK(vec && len == 4 && vec[0] == 9.0f, "vector cell");
    flats_result_free(res);

    CHECK(flats_execute(db, "SELEC", NULL) == FLATS_ERR_QUERY, "bad sql is a query error");
    CHECK(flats_last_error() != NULL, "with a message");

    CHECK(flats_backup(db, argv[2]) == FLATS_OK, "backup");
    CHECK(flats_close(db) == FLATS_OK, "close");

    flats_db *copy = NULL;
    CHECK(flats_open(argv[2], &copy) == FLATS_OK, "open the backup");
    CHECK(flats_execute(copy, "SELECT id FROM docs;", &res) == FLATS_OK, "select from it");
    CHECK(flats_result_row_count(res) == 10, "every row is there");
    flats_result_free(res);
    CHECK(flats_close(copy) == FLATS_OK, "close the backup");
    return 0;
}
