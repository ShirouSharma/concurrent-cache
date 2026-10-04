//! The public-facing sharded concurrent cache.
//!
//! # Concurrency design
//!
//! ```text
//!     Thread A                Thread B               Thread C
//!        │                       │                       │
//!        │ insert("foo", 1)      │ get("bar")            │ insert("baz", 3)
//!        │                       │                       │
//!        ▼                       ▼                       ▼
//!   hash("foo") % N = 2    hash("bar") % N = 5     hash("baz") % N = 2
//!        │                       │                       │
//!        ▼                       ▼                       ▼
//!   write_lock(shard[2])    write_lock(shard[5])    write_lock(shard[2])
//!        │                       │                       │
//!        │ ← NO CONTENTION →     │      ← CONTENDS →    │
//!        │   (different shards)   │   (same shard — B     │
//!        │                       │    must wait for A)    │
//! ```
//!
//! Threads hitting **different shards** proceed in parallel with zero contention.
//! Threads hitting the **same shard** serialize on that shard's `RwLock`.
//! More shards = less contention = better throughput (at the cost of memory overhead).
//!
//! # Why `get()` takes a write lock
//!
//! In an LRU cache, every read updates the eviction ordering (promotion to front).
//! This means `get()` is a **mutating** operation on the shard, requiring a write lock.
//! A read lock would only work for `peek()` and `contains()`, which don't touch the
//! LRU list.
//!
//! # Poison recovery
//!
//! If a thread panics while holding a lock, the `RwLock` becomes "poisoned". We
//! recover from poisoning via `unwrap_or_else(|e| e.into_inner())` — the shard
//! data might be slightly inconsistent, but for a cache, serving stale data is
//! preferable to crashing all threads. This is a deliberate choice.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use crate::shard::Shard;

/// Default number of shards. Higher = less contention, more memory overhead.
/// 16 is a common default (used by DashMap, Java ConcurrentHashMap pre-8, etc.)
const DEFAULT_SHARD_COUNT: usize = 16;

/// Snapshot of cache performance statistics.
///
/// Hit/miss counts use `Relaxed` ordering — they're approximate under concurrent
/// access but never torn (atomics guarantee this). Good enough for monitoring;
/// not suitable for billing.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of `get()` calls that found the key.
    pub hits: usize,

    /// Number of `get()` calls that did NOT find the key.
    pub misses: usize,

    /// Current number of entries across all shards.
    pub len: usize,
}

impl CacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`. Returns 0.0 if no operations yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CacheStats {{ hits: {}, misses: {}, len: {}, hit_rate: {:.1}% }}",
            self.hits,
            self.misses,
            self.len,
            self.hit_rate() * 100.0,
        )
    }
}

/// A thread-safe, sharded LRU cache.
///
/// The total capacity is distributed evenly across `N` shards. Each shard is
/// independently locked, so concurrent operations on different shards proceed
/// without contention.
///
/// # Type parameters
///
/// - `K`: Key type. Must be `Hash + Eq + Clone`.
///   - `Hash + Eq` for HashMap lookups.
///   - `Clone` because the key is stored in both the HashMap and the LRU list node.
/// - `V`: Value type. Must be `Clone` because `get()` returns an owned copy
///   (the lock is released before returning, so we can't return a reference).
/// - `S`: Hash builder (defaults to `RandomState`). Pluggable for deterministic
///   testing or custom hashers.
pub struct ConcurrentCache<K, V, S = RandomState> {
    /// The shard array. `Box<[...]>` instead of `Vec` because it never resizes.
    shards: Box<[RwLock<Shard<K, V>>]>,

    /// Number of shards (cached to avoid `.len()` on the slice repeatedly).
    num_shards: usize,

    /// Hasher factory — used to hash keys for shard routing.
    hash_builder: S,

    /// Total cache hits (across all shards). Updated with `Relaxed` ordering.
    hits: AtomicUsize,

    /// Total cache misses (across all shards). Updated with `Relaxed` ordering.
    misses: AtomicUsize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructors (only available with the default RandomState hasher)
// ──────────────────────────────────────────────────────────────────────────────
impl<K, V> ConcurrentCache<K, V, RandomState>
where
    K: Hash + Eq + Clone,
{
    /// Create a new cache with `capacity` total entries and the default shard count.
    ///
    /// # Example
    /// ```
    /// use concurrent_cache::ConcurrentCache;
    /// let cache: ConcurrentCache<String, i32> = ConcurrentCache::new(1000);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::with_shards(capacity, DEFAULT_SHARD_COUNT)
    }

    /// Create a new cache with explicit shard count.
    ///
    /// `capacity` is divided evenly across shards (rounded up). More shards
    /// reduce contention but increase memory overhead.
    ///
    /// # Panics
    /// Panics if `num_shards` is 0.
    ///
    /// # Example
    /// ```
    /// use concurrent_cache::ConcurrentCache;
    /// // 4 shards, 250 entries each
    /// let cache: ConcurrentCache<String, i32> = ConcurrentCache::with_shards(1000, 4);
    /// ```
    pub fn with_shards(capacity: usize, num_shards: usize) -> Self {
        Self::with_hasher_and_shards(capacity, num_shards, RandomState::new())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core operations (generic over hasher S)
// ──────────────────────────────────────────────────────────────────────────────
impl<K, V, S> ConcurrentCache<K, V, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher,
{
    /// Full constructor with custom hasher and shard count.
    pub fn with_hasher_and_shards(capacity: usize, num_shards: usize, hash_builder: S) -> Self {
        assert!(num_shards > 0, "cache requires at least one shard");

        // Ceiling division: distribute capacity evenly, rounding up so we
        // never lose capacity. Example: capacity=10, shards=3 → 4 per shard.
        let per_shard = if capacity == 0 {
            0
        } else {
            (capacity + num_shards - 1) / num_shards
        };

        let shards: Vec<_> = (0..num_shards)
            .map(|_| RwLock::new(Shard::new(per_shard)))
            .collect();

        Self {
            shards: shards.into_boxed_slice(),
            num_shards,
            hash_builder,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Compute which shard a key belongs to.
    ///
    /// We hash the key once and use the hash to select a shard via modulo.
    /// The same hash is reused by the shard's inner HashMap (since HashMap
    /// hashes the key again with its own hasher — this is acceptable overhead).
    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.num_shards
    }

    /// Retrieve a value by key, promoting it in the LRU ordering.
    ///
    /// Returns `Some(value)` if found (cloned from under the lock), or `None`.
    ///
    /// # Lock behavior
    /// Takes a **write lock** on the target shard (because LRU promotion mutates
    /// the list). If you don't need promotion, use [`peek`](Self::peek) which
    /// only takes a read lock.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write().unwrap_or_else(|e| e.into_inner());

        match shard.get(key) {
            Some(v) => {
                let cloned = v.clone();
                // Relaxed ordering: we don't need other threads to see this
                // increment immediately. Stats are approximate by design.
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(cloned)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Retrieve a value **without** promoting it in the LRU ordering.
    ///
    /// # Lock behavior
    /// Takes a **read lock** — multiple threads can `peek` the same shard
    /// concurrently without blocking each other.
    pub fn peek(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read().unwrap_or_else(|e| e.into_inner());
        shard.peek(key).cloned()
    }

    /// Insert a key-value pair into the cache.
    ///
    /// - If the key already exists, the value is updated and the old value is returned.
    /// - If the shard is at capacity, the least-recently-used entry is evicted silently.
    ///
    /// # Lock behavior
    /// Takes a **write lock** on the target shard.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx].write().unwrap_or_else(|e| e.into_inner());
        shard.insert(key, value)
    }

    /// Remove a key from the cache. Returns `Some(value)` if the key was found.
    ///
    /// # Lock behavior
    /// Takes a **write lock** on the target shard.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write().unwrap_or_else(|e| e.into_inner());
        shard.remove(key)
    }

    /// Check if a key exists without promoting it and without cloning the value.
    ///
    /// # Lock behavior
    /// Takes a **read lock** on the target shard.
    pub fn contains(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read().unwrap_or_else(|e| e.into_inner());
        shard.contains(key)
    }

    /// Total number of entries across all shards.
    ///
    /// Note: this acquires a **read lock** on every shard sequentially, so the
    /// count may be slightly stale by the time it's returned. Don't use this for
    /// synchronization — it's for monitoring only.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().unwrap_or_else(|e| e.into_inner()).len())
            .sum()
    }

    /// Returns `true` if the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|s| s.read().unwrap_or_else(|e| e.into_inner()).len() == 0)
    }

    /// Snapshot of hit/miss statistics and current size.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            len: self.len(),
        }
    }

    /// Number of shards the cache is split into.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Remove all entries from the cache.
    ///
    /// Acquires a write lock on each shard sequentially and replaces each
    /// with a fresh, empty shard of the same capacity.
    pub fn clear(&self) {
        for shard_lock in self.shards.iter() {
            let mut shard = shard_lock.write().unwrap_or_else(|e| e.into_inner());
            shard.clear();
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Additional impls
// ──────────────────────────────────────────────────────────────────────────────
impl<K, V, S> std::fmt::Debug for ConcurrentCache<K, V, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcurrentCache")
            .field("num_shards", &self.num_shards)
            .field("len", &self.len())
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .finish()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests (single-threaded)
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Use a single shard for deterministic testing (no hash-based routing variance).
    fn single_shard_cache(capacity: usize) -> ConcurrentCache<&'static str, i32> {
        ConcurrentCache::with_shards(capacity, 1)
    }

    #[test]
    fn basic_insert_get_remove() {
        let cache = single_shard_cache(10);
        assert!(cache.insert("a", 1).is_none());
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.remove(&"a"), Some(1));
        assert!(cache.get(&"a").is_none());
    }

    #[test]
    fn update_returns_old_value() {
        let cache = single_shard_cache(10);
        cache.insert("a", 1);
        let old = cache.insert("a", 42);
        assert_eq!(old, Some(1));
        assert_eq!(cache.get(&"a"), Some(42));
    }

    #[test]
    fn eviction_respects_lru() {
        let cache = single_shard_cache(3);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        // Access "a" to promote it
        cache.get(&"a");

        // Insert "d" — should evict "b" (now LRU)
        cache.insert("d", 4);
        assert!(cache.get(&"b").is_none());
        assert_eq!(cache.get(&"a"), Some(1));
    }

    #[test]
    fn len_and_is_empty() {
        let cache = single_shard_cache(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
    }

    #[test]
    fn contains_does_not_promote() {
        let cache = single_shard_cache(2);
        cache.insert("a", 1); // LRU
        cache.insert("b", 2); // MRU

        // contains does not promote
        assert!(cache.contains(&"a"));

        // Insert "c" — should evict "a" (still LRU since contains didn't promote)
        cache.insert("c", 3);
        assert!(!cache.contains(&"a"));
    }

    #[test]
    fn peek_does_not_promote() {
        let cache = single_shard_cache(2);
        cache.insert("a", 1); // LRU
        cache.insert("b", 2); // MRU

        // peek returns value but doesn't promote
        assert_eq!(cache.peek(&"a"), Some(1));

        // Insert "c" — should evict "a"
        cache.insert("c", 3);
        assert!(cache.peek(&"a").is_none());
    }

    #[test]
    fn stats_tracking() {
        let cache = single_shard_cache(10);
        cache.insert("a", 1);

        cache.get(&"a"); // hit
        cache.get(&"a"); // hit
        cache.get(&"b"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn stats_display() {
        let cache = single_shard_cache(10);
        cache.insert("a", 1);
        cache.get(&"a");
        cache.get(&"b");
        let stats = cache.stats();
        let s = format!("{}", stats);
        assert!(s.contains("hits: 1"));
        assert!(s.contains("misses: 1"));
    }

    #[test]
    fn debug_impl() {
        let cache = single_shard_cache(10);
        cache.insert("a", 1);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("ConcurrentCache"));
        assert!(dbg.contains("num_shards"));
    }

    #[test]
    fn multi_shard_basic() {
        let cache: ConcurrentCache<i32, i32> = ConcurrentCache::with_shards(100, 4);
        for i in 0..50 {
            cache.insert(i, i * 10);
        }
        assert_eq!(cache.len(), 50);
        for i in 0..50 {
            assert_eq!(cache.get(&i), Some(i * 10));
        }
    }

    #[test]
    fn num_shards_accessor() {
        let cache: ConcurrentCache<i32, i32> = ConcurrentCache::with_shards(100, 8);
        assert_eq!(cache.num_shards(), 8);
    }

    #[test]
    #[should_panic(expected = "at least one shard")]
    fn zero_shards_panics() {
        let _cache: ConcurrentCache<i32, i32> = ConcurrentCache::with_shards(100, 0);
    }
}
