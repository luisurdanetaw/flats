// Type declarations for the Flats Node.js bindings.

export type Cell = number | bigint | string | number[];

export class Result implements Iterable<Cell[]> {
  readonly columns: string[];
  readonly rows: Cell[][];
  readonly length: number;
  [Symbol.iterator](): Iterator<Cell[]>;
  toObjects(): Record<string, Cell>[];
}

export type Row = Record<string, number | bigint | string | number[] | Float32Array | Float64Array>;

/** A value for a `?` placeholder. */
export type Param = number | bigint | string | ArrayLike<number>;

export interface SearchOptions {
  k?: number;
  /** A V-SQL predicate; put values in `params` as `?`. */
  where?: string;
  params?: Param[];
  returning?: string[];
}

export class RowIterator implements IterableIterator<Cell[]> {
  readonly columns: string[];
  readonly closed: boolean;
  next(): IteratorResult<Cell[]>;
  return(): IteratorResult<Cell[]>;
  close(): void;
  [Symbol.iterator](): RowIterator;
}

export class Database {
  constructor(path: string);
  readonly closed: boolean;
  close(): void;
  execute(sql: string, params?: Param[]): Result;
  iterate(sql: string, params?: Param[], options?: { chunkRows?: number }): RowIterator;
  search(collection: string, vector: ArrayLike<number>, options?: SearchOptions): Result;
  insert(collection: string, rows: Iterable<Row>): number;
  backup(dest: string): void;
}

/** Rows of one statement, read lazily on the thread pool: use `for await`. */
export class AsyncRowIterator implements AsyncIterableIterator<Cell[]> {
  readonly columns: string[];
  readonly closed: boolean;
  next(): Promise<IteratorResult<Cell[]>>;
  return(): Promise<IteratorResult<Cell[]>>;
  close(): void;
  [Symbol.asyncIterator](): AsyncRowIterator;
}

/** The Promise API: every call runs on libuv's thread pool. */
export class AsyncDatabase {
  readonly closed: boolean;
  /** Waits for calls in flight, then checkpoints and closes. */
  close(): Promise<void>;
  execute(sql: string, params?: Param[]): Promise<Result>;
  query(sql: string, params?: Param[], options?: { chunkRows?: number }): Promise<AsyncRowIterator>;
  search(collection: string, vector: ArrayLike<number>, options?: SearchOptions): Promise<Result>;
  insert(collection: string, rows: Iterable<Row>): Promise<number>;
  backup(dest: string): Promise<void>;
}

/** Open (creating if missing) a database; resolves to an AsyncDatabase. */
export function open(path: string): Promise<AsyncDatabase>;

export class FlatsError extends Error {
  readonly code: number;
}
export class InvalidArgumentError extends FlatsError {}
export class QueryError extends FlatsError {}
export class EngineError extends FlatsError {}
export class InternalError extends FlatsError {}

export const BATCH_ROWS: number;
export function version(): string;
