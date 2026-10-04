use concurrent_cache::ConcurrentCache;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Concurrent Sharded LRU Cache — Demo              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Create a cache shared across threads ──
    let cache = Arc::new(ConcurrentCache::new(500));
    println!("Created cache: capacity=500, shards={}", cache.num_shards());
    println!();

    // ── 2. Concurrent writes from 8 threads ──
    println!("▸ Spawning 8 writer threads (250 inserts each)...");
    let mut handles = vec![];
    for t in 0..8 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..250 {
                let key = t * 250 + i;
                cache.insert(key, format!("thread{}-val{}", t, i));
            }
        }));
    }

    for h in handles {
        h.join().expect("writer thread panicked");
    }

    println!(
        "  ✓ All writers done. Cache size: {} (capped at 500)",
        cache.len()
    );
    println!();

    // ── 3. Read some values ──
    println!("▸ Reading keys 0..10:");
    for i in 0..10 {
        match cache.get(&i) {
            Some(v) => println!("  key {:>3} → {}", i, v),
            None => println!("  key {:>3} → (evicted)", i),
        }
    }
    println!();

    // ── 4. Concurrent reads + writes ──
    println!("▸ Spawning 4 reader + 4 writer threads for mixed workload...");
    let mut handles = vec![];

    // Readers
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            let mut found = 0;
            for i in 0..500 {
                if cache.get(&i).is_some() {
                    found += 1;
                }
            }
            found
        }));
    }

    // Writers (updating existing keys)
    for t in 0..4 {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..500 {
                cache.insert(i, format!("updated-t{}-{}", t, i));
            }
            0usize // placeholder return
        }));
    }

    for h in handles {
        h.join().expect("mixed thread panicked");
    }

    println!("  ✓ Mixed workload complete.");
    println!();

    // ── 5. Stats ──
    let stats = cache.stats();
    println!("▸ Final statistics:");
    println!("  {}", stats);
    println!();

    // ── 6. Demonstrate peek vs get ──
    println!("▸ peek() vs get() demonstration:");
    cache.insert(9999, "peeked".to_string());
    println!("  peek(9999) = {:?}  (no LRU promotion)", cache.peek(&9999));
    println!("  get(9999)  = {:?}  (promoted to MRU)", cache.get(&9999));
    println!();

    println!("Done! No panics, no data races. 🎉");
}
