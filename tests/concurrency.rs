//! Concurrency stress tests.
//!
//! These tests spawn many threads doing concurrent operations to verify:
//! - No panics (data corruption)
//! - No deadlocks (tests time out if deadlocked)
//! - Correct value integrity (values match expected patterns)
//! - Stat counters are accurate

use concurrent_cache::ConcurrentCache;
use std::sync::Arc;
use std::thread;

#[test]
fn stress_concurrent_insert_and_get() {
    let cache = Arc::new(ConcurrentCache::new(500));
    let num_threads = 8;
    let ops_per_thread = 1000;

    let mut handles = vec![];

    // Spawn writer threads
    for t in 0..num_threads {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = t * ops_per_thread + i;
                cache.insert(key, key * 10);
            }
        }));
    }

    // Spawn reader threads concurrently
    for t in 0..num_threads {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let key = t * ops_per_thread + i;
                // Value may or may not exist yet — that's fine
                if let Some(v) = cache.get(&key) {
                    assert_eq!(v, key * 10, "value mismatch for key {}", key);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked — potential data corruption");
    }

    // Cache len may slightly exceed the requested capacity because per-shard
    // capacity is rounded up (ceiling division). With 16 shards and 500 total,
    // each shard gets ceil(500/16) = 32, so actual max = 32×16 = 512.
    let max_possible = {
        let shards = 16usize;
        let per_shard = (500 + shards - 1) / shards;
        per_shard * shards
    };
    assert!(
        cache.len() <= max_possible,
        "cache exceeded max capacity {}: got {}",
        max_possible,
        cache.len()
    );
}

#[test]
fn stress_mixed_operations() {
    let cache = Arc::new(ConcurrentCache::new(200));
    let mut handles = vec![];

    for t in 0..12 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..500 {
                let key = (t * 500 + i) % 300; // overlapping key space
                match i % 4 {
                    0 => {
                        cache.insert(key, format!("t{}-{}", t, i));
                    }
                    1 => {
                        let _ = cache.get(&key);
                    }
                    2 => {
                        let _ = cache.remove(&key);
                    }
                    _ => {
                        let _ = cache.contains(&key);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during mixed operations");
    }

    // Just verify we survived without panics or deadlocks
    let stats = cache.stats();
    println!(
        "Mixed stress: len={}, hits={}, misses={}, hit_rate={:.1}%",
        stats.len,
        stats.hits,
        stats.misses,
        stats.hit_rate() * 100.0
    );
}

#[test]
fn stress_high_contention_single_key() {
    // Many threads fighting over the SAME key — maximum lock contention
    let cache = Arc::new(ConcurrentCache::new(10));
    let mut handles = vec![];

    for t in 0..16 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..1000 {
                cache.insert(0u64, t * 1000 + i);
                let _ = cache.get(&0u64);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked under high contention");
    }

    // Key 0 should exist with SOME value (we can't predict which thread wrote last)
    assert!(cache.get(&0u64).is_some(), "key 0 should exist");
}

#[test]
fn stats_consistency_under_concurrency() {
    // Use a single shard with capacity 100 so all 50 pre-populated keys fit
    // without any eviction due to hash distribution imbalance.
    let cache = Arc::new(ConcurrentCache::with_shards(100, 1));

    // Pre-populate with keys 0..50
    for i in 0..50u32 {
        cache.insert(i, i);
    }
    assert_eq!(cache.len(), 50, "pre-population should insert exactly 50 keys");

    let mut handles = vec![];
    let threads = 8;
    let ops = 100;

    for _ in 0..threads {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..ops {
                // Keys 0..49 should hit, 50..99 should miss
                let _ = cache.get(&(i as u32));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let stats = cache.stats();
    let total = stats.hits + stats.misses;

    // Total operations = threads × ops = 800
    assert_eq!(total, threads * ops, "total ops mismatch");

    // Each thread does 50 hits + 50 misses (no eviction since we only read)
    assert_eq!(stats.hits, threads * 50, "hit count mismatch");
    assert_eq!(stats.misses, threads * 50, "miss count mismatch");
}

#[test]
fn concurrent_insert_remove_same_keys() {
    // Half the threads insert, half remove — on overlapping keys
    let cache = Arc::new(ConcurrentCache::new(100));
    let mut handles = vec![];

    // Inserters
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..200u32 {
                cache.insert(i, i);
            }
        }));
    }

    // Removers
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..200u32 {
                let _ = cache.remove(&i);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during insert/remove race");
    }

    // We can't predict the exact final state, but it should be consistent
    // Cache len may slightly exceed the requested capacity because per-shard
    // capacity is rounded up (ceiling division).
    let max_possible = {
        let shards = 16usize;
        let per_shard = (100 + shards - 1) / shards;
        per_shard * shards
    };
    let len = cache.len();
    assert!(
        len <= max_possible,
        "cache size {} exceeds max capacity {}",
        len,
        max_possible
    );
}

#[test]
fn peek_under_concurrent_writes() {
    let cache = Arc::new(ConcurrentCache::new(100));

    // Pre-populate
    for i in 0..100u32 {
        cache.insert(i, i * 10);
    }

    let mut handles = vec![];

    // Peekers (read lock only)
    for _ in 0..8 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                if let Some(v) = cache.peek(&i) {
                    // Value should be either original or updated
                    assert!(
                        v == i * 10 || v == i * 100,
                        "unexpected value {} for key {}",
                        v,
                        i
                    );
                }
            }
        }));
    }

    // Writers (write lock — updating values)
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                cache.insert(i, i * 100);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread panicked during peek + write");
    }
}
