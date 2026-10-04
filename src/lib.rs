//! # Concurrent Cache
//!
//! A thread-safe, sharded LRU cache built entirely from Rust standard library
//! primitives. Designed as a learning exercise for Rust concurrency patterns.
//!
//! ## Architecture
//!
//! The cache splits its keyspace across **N independent shards**, each protected
//! by its own [`std::sync::RwLock`]. Keys are routed to shards via hashing, so
//! most operations only contend on a single shard — dramatically reducing lock
//! contention under concurrent access.
//!
//! Each shard maintains:
//! - A [`std::collections::HashMap`] for O(1) key lookups
//! - An arena-based doubly-linked list for LRU ordering
//!
//! ## Concurrency Primitives Used
//!
//! | Primitive | Purpose |
//! |---|---|
//! | `RwLock` | Per-shard reader/writer lock |
//! | `AtomicUsize` | Lock-free hit/miss counters |
//! | `HashMap` | O(1) key→node mapping |
//! | Arena `Vec` | Cache-friendly node storage |
//!
//! ## Quick Start
//!
//! ```
//! use concurrent_cache::ConcurrentCache;
//!
//! let cache = ConcurrentCache::new(100);
//! cache.insert("hello", 42);
//! assert_eq!(cache.get(&"hello"), Some(42));
//! ```

mod cache;
mod list;
mod shard;

pub use cache::{CacheStats, ConcurrentCache};
