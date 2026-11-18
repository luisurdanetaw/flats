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

export interface SearchOptions {
  k?: number;
  where?: string;
  returning?: string[];
}

export class Database {
  constructor(path: string);
  readonly closed: boolean;
  close(): void;
  execute(sql: string): Result;
  search(collection: string, vector: ArrayLike<number>, options?: SearchOptions): Result;
  insert(collection: string, rows: Iterable<Row>): number;
  backup(dest: string): void;
}

export class FlatsError extends Error {
  readonly code: number;
}
export class InvalidArgumentError extends FlatsError {}
export class QueryError extends FlatsError {}
export class EngineError extends FlatsError {}
export class InternalError extends FlatsError {}

export const BATCH_ROWS: number;
export function version(): string;
