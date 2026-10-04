//! A single cache shard: `HashMap` + `LruList` working together.
//!
//! # How it works
//!
//! The `HashMap<K, usize>` maps keys to their arena index in the `LruList`.
//! This gives us:
//! - **O(1) lookup**: hash the key → get arena index → read the node
//! - **O(1) promotion**: move the accessed node to the front of the LRU list
//! - **O(1) eviction**: pop the tail of the LRU list, remove from HashMap
//!
//! ```text
//!   HashMap                 LruList (doubly-linked, arena-backed)
//!  ┌────────┐
//!  │ "foo"→2│───────┐      head ──→ [2: "foo"] ←→ [0: "bar"] ←→ [1: "baz"] ←── tail
//!  │ "bar"→0│───┐   │             (most recent)                  (least recent)
//!  │ "baz"→1│─┐ │   │
//!  └────────┘ │ │   │
//!             └─┘───┘  (indices point into the arena Vec)
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use crate::list::LruList;

/// A single shard of the cache. Not thread-safe on its own — the outer
/// `ConcurrentCache` wraps each shard in an `RwLock`.
pub(crate) struct Shard<K, V> {
    /// Maps keys to their arena index in `list`.
    map: HashMap<K, usize>,

    /// Doubly-linked list maintaining LRU order.
    list: LruList<K, V>,

    /// Maximum number of entries this shard can hold before eviction.
    capacity: usize,
}

impl<K, V> Shard<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create a new shard with the given per-shard capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            list: LruList::with_capacity(capacity),
            capacity,
        }
    }

    /// Look up a key and **promote** it to the front of the LRU list.
    ///
    /// Returns a reference to the value, or `None` if the key isn't present.
    ///
    /// # Why `&mut self`?
    ///
    /// Even though this is logically a "read", it mutates the LRU ordering.
    /// At the `ConcurrentCache` level, this means `get()` requires a **write lock**
    /// on the shard — a fundamental trade-off in LRU caches. The alternative
    /// (read lock + deferred promotion) adds complexity for marginal gain.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            self.list.move_to_front(idx);
            Some(self.list.get(idx))
        } else {
            None
        }
    }

    /// Look up a key **without** promoting it in the LRU list.
    ///
    /// Takes `&self` — at the cache level, this only needs a **read lock**.
    /// Useful for checking values without disturbing eviction ordering.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|&idx| self.list.get(idx))
    }

    /// Insert a key-value pair, evicting the LRU entry if at capacity.
    ///
    /// - If `key` already exists: updates the value in-place and promotes it.
    ///   Returns `Some(old_value)`.
    /// - If `key` is new and the shard is full: evicts the tail (LRU) entry
    ///   silently, then inserts. Returns `None`.
    /// - If `key` is new and there's room: inserts directly. Returns `None`.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // ── Case 1: Key already exists → update in place ──
        if let Some(&idx) = self.map.get(&key) {
            let old_value = std::mem::replace(self.list.get_mut(idx), value);
            self.list.move_to_front(idx);
            return Some(old_value);
        }

        // ── Case 2: At capacity → evict LRU (tail) entry ──
        if self.capacity > 0 && self.list.len() >= self.capacity {
            if let Some((evicted_key, _evicted_value)) = self.list.pop_back() {
                self.map.remove(&evicted_key);
            }
        }

        // ── Case 3: Insert new entry at front ──
        if self.capacity == 0 {
            // Degenerate case: zero-capacity shard stores nothing.
            return None;
        }

        let idx = self.list.push_front(key.clone(), value);
        self.map.insert(key, idx);

        None
    }

    /// Remove a key explicitly. Returns `Some(value)` if found.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key).map(|idx| {
            let (_k, v) = self.list.remove(idx);
            v
        })
    }

    /// Check if a key exists without promoting it.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Number of entries in this shard.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// The maximum capacity of this shard.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove all entries, resetting the shard to an empty state.
    pub fn clear(&mut self) {
        self.map.clear();
        self.list = LruList::with_capacity(self.capacity);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_and_get() {
        let mut shard = Shard::new(4);
        assert!(shard.insert("a", 1).is_none());
        assert_eq!(shard.get(&"a"), Some(&1));
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn update_existing_key() {
        let mut shard = Shard::new(4);
        shard.insert("a", 1);

        // Updating should return the old value
        let old = shard.insert("a", 42);
        assert_eq!(old, Some(1));
        assert_eq!(shard.get(&"a"), Some(&42));
        assert_eq!(shard.len(), 1); // no new entry, just updated
    }

    #[test]
    fn eviction_at_capacity() {
        let mut shard = Shard::new(3);
        shard.insert("a", 1); // LRU
        shard.insert("b", 2);
        shard.insert("c", 3); // MRU

        // At capacity — inserting "d" should evict "a" (LRU)
        shard.insert("d", 4);
        assert_eq!(shard.len(), 3);
        assert!(shard.get(&"a").is_none()); // evicted
        assert_eq!(shard.get(&"d"), Some(&4)); // new entry present
    }

    #[test]
    fn access_prevents_eviction() {
        let mut shard = Shard::new(3);
        shard.insert("a", 1);
        shard.insert("b", 2);
        shard.insert("c", 3);

        // Access "a" — promotes it to MRU, so "b" becomes LRU
        shard.get(&"a");

        // Inserting "d" should now evict "b" (not "a")
        shard.insert("d", 4);
        assert!(shard.get(&"b").is_none()); // evicted
        assert_eq!(shard.get(&"a"), Some(&1)); // still alive — was promoted
    }

    #[test]
    fn remove_entry() {
        let mut shard = Shard::new(4);
        shard.insert("a", 1);
        shard.insert("b", 2);

        let removed = shard.remove(&"a");
        assert_eq!(removed, Some(1));
        assert!(shard.get(&"a").is_none());
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn remove_nonexistent() {
        let mut shard: Shard<&str, i32> = Shard::new(4);
        assert!(shard.remove(&"ghost").is_none());
    }

    #[test]
    fn contains_check() {
        let mut shard = Shard::new(4);
        shard.insert("a", 1);
        assert!(shard.contains(&"a"));
        assert!(!shard.contains(&"b"));
    }

    #[test]
    fn peek_does_not_promote() {
        let mut shard = Shard::new(2);
        shard.insert("a", 1); // LRU
        shard.insert("b", 2); // MRU

        // Peek at "a" — should NOT promote it
        assert_eq!(shard.peek(&"a"), Some(&1));

        // Insert "c" — should evict "a" (still LRU since peek didn't promote)
        shard.insert("c", 3);
        assert!(shard.peek(&"a").is_none());
        assert_eq!(shard.peek(&"b"), Some(&2));
    }

    #[test]
    fn capacity_one() {
        let mut shard = Shard::new(1);
        shard.insert("a", 1);
        assert_eq!(shard.get(&"a"), Some(&1));

        // Inserting a second key evicts the first
        shard.insert("b", 2);
        assert!(shard.get(&"a").is_none());
        assert_eq!(shard.get(&"b"), Some(&2));
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn zero_capacity() {
        let mut shard: Shard<&str, i32> = Shard::new(0);
        shard.insert("a", 1);
        // Nothing stored
        assert_eq!(shard.len(), 0);
        assert!(shard.get(&"a").is_none());
    }

    #[test]
    fn eviction_order_with_multiple_accesses() {
        let mut shard = Shard::new(3);
        shard.insert("a", 1);
        shard.insert("b", 2);
        shard.insert("c", 3);

        // Access pattern: get a, get b → order becomes: b(MRU) → a → c(LRU)
        shard.get(&"a");
        shard.get(&"b");

        // Evict: should remove "c" (LRU)
        shard.insert("d", 4);
        assert!(shard.get(&"c").is_none());
        assert!(shard.contains(&"a"));
        assert!(shard.contains(&"b"));
        assert!(shard.contains(&"d"));
    }
}
