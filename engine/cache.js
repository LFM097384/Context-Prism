// Small in-memory TTL + LRU cache.

export class TTLCache {
  constructor({ maxsize = 128, ttlSeconds = 60 } = {}) {
    this.maxsize = maxsize;
    this.ttlSeconds = ttlSeconds;
    this._data = new Map();
    this._expires = new Map();
  }

  get(key) {
    if (!this._data.has(key)) return undefined;
    if (Date.now() / 1000 > (this._expires.get(key) || 0)) {
      this._drop(key);
      return undefined;
    }
    const value = this._data.get(key);
    this._data.delete(key);
    this._data.set(key, value);
    return value;
  }

  set(key, value) {
    this._data.delete(key);
    this._data.set(key, value);
    this._expires.set(key, Date.now() / 1000 + this.ttlSeconds);
    while (this._data.size > this.maxsize) {
      const oldest = this._data.keys().next().value;
      this._drop(oldest);
    }
  }

  clear() {
    this._data.clear();
    this._expires.clear();
  }

  get size() {
    return this._data.size;
  }

  _drop(key) {
    this._data.delete(key);
    this._expires.delete(key);
  }
}
