//! Arena-based intrusive doubly-linked list for LRU ordering.
//!
//! # Why arena allocation?
//!
//! A classic doubly-linked list in Rust using `Box<Node>` and raw pointers is
//! notoriously hard to get right safely. Instead, we store all nodes in a
//! contiguous `Vec<Option<Node>>` (the "arena") and refer to them by index.
//!
//! This gives us:
//! - **O(1) insert/remove/move** — same as a pointer-based list
//! - **Cache-friendly memory layout** — nodes live in a contiguous `Vec`
//! - **No `unsafe`** — indices are just `usize`, no raw pointer juggling
//! - **Free-list reuse** — removed slots are recycled, avoiding unbounded growth
//!
//! # Trade-offs
//!
//! - Indices can dangle if misused (but this is `pub(crate)`, so only we use it)
//! - The arena `Vec` never shrinks (acceptable for a fixed-capacity cache)

/// A single node in the linked list, stored inside the arena.
///
/// Each node holds a key-value pair and pointers (indices) to its
/// neighbors in the LRU chain.
pub(crate) struct Node<K, V> {
    pub key: K,
    pub value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Arena-based doubly-linked list.
///
/// Nodes are stored in `slots` (a `Vec<Option<Node>>`). Live nodes are `Some`,
/// removed nodes are `None` and their indices sit on `free_indices` for reuse.
///
/// `head` points to the most-recently-used node (front of list).
/// `tail` points to the least-recently-used node (back of list).
pub(crate) struct LruList<K, V> {
    /// The arena — stores all nodes, indexed by `usize`.
    slots: Vec<Option<Node<K, V>>>,

    /// Index of the most-recently-used node (front).
    head: Option<usize>,

    /// Index of the least-recently-used node (back).
    tail: Option<usize>,

    /// Recycled slot indices from removed nodes — avoids allocating new slots
    /// when the arena already has holes.
    free_indices: Vec<usize>,

    /// Number of live nodes. Tracked explicitly because `slots.len()` includes
    /// dead (None) slots.
    len: usize,
}

impl<K, V> LruList<K, V> {
    /// Create an empty list with pre-allocated arena capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            head: None,
            tail: None,
            free_indices: Vec::new(),
            len: 0,
        }
    }

    /// Allocate a slot for `node`, reusing a free index if available.
    /// Returns the index where the node was placed.
    fn allocate(&mut self, node: Node<K, V>) -> usize {
        if let Some(idx) = self.free_indices.pop() {
            // Reuse a previously freed slot
            self.slots[idx] = Some(node);
            idx
        } else {
            // Grow the arena
            let idx = self.slots.len();
            self.slots.push(Some(node));
            idx
        }
    }

    /// Insert a new key-value pair at the **front** (most-recently-used position).
    ///
    /// Returns the arena index of the new node — the caller (Shard) stores this
    /// in its HashMap for O(1) future lookups.
    pub fn push_front(&mut self, key: K, value: V) -> usize {
        let node = Node {
            key,
            value,
            prev: None,
            next: self.head, // new node's next → old head
        };
        let idx = self.allocate(node);

        // Old head's prev now points back to the new node
        if let Some(old_head) = self.head {
            self.slots[old_head].as_mut().unwrap().prev = Some(idx);
        }

        self.head = Some(idx);

        // If the list was empty, this node is also the tail
        if self.tail.is_none() {
            self.tail = Some(idx);
        }

        self.len += 1;
        idx
    }

    /// Move an existing node to the **front** of the list (LRU promotion).
    ///
    /// This is the core LRU operation: every time a key is accessed, its node
    /// is promoted to the front so the tail always holds the least-recently-used
    /// entry.
    ///
    /// Runs in O(1) — just pointer (index) swaps, no data movement.
    pub fn move_to_front(&mut self, idx: usize) {
        // Already at front — nothing to do
        if self.head == Some(idx) {
            return;
        }

        // Step 1: Unlink from current position
        let (prev, next) = {
            let node = self.slots[idx].as_ref().unwrap();
            (node.prev, node.next)
        };

        // Stitch neighbors together
        if let Some(prev_idx) = prev {
            self.slots[prev_idx].as_mut().unwrap().next = next;
        }
        // (no else: if prev is None, the node was head — but we checked that above)

        if let Some(next_idx) = next {
            self.slots[next_idx].as_mut().unwrap().prev = prev;
        }

        // If this node was the tail, update tail to the predecessor
        if self.tail == Some(idx) {
            self.tail = prev;
        }

        // Step 2: Link at front
        {
            let node = self.slots[idx].as_mut().unwrap();
            node.prev = None;
            node.next = self.head;
        }

        if let Some(old_head) = self.head {
            self.slots[old_head].as_mut().unwrap().prev = Some(idx);
        }

        self.head = Some(idx);
    }

    /// Remove a node by arena index. Returns the key-value pair.
    ///
    /// The freed slot is added to `free_indices` for future reuse.
    pub fn remove(&mut self, idx: usize) -> (K, V) {
        let node = self.slots[idx].take().expect("BUG: removing an empty slot");

        // Unlink: stitch prev ↔ next
        match node.prev {
            Some(prev_idx) => {
                self.slots[prev_idx].as_mut().unwrap().next = node.next;
            }
            None => {
                // This node was the head
                self.head = node.next;
            }
        }

        match node.next {
            Some(next_idx) => {
                self.slots[next_idx].as_mut().unwrap().prev = node.prev;
            }
            None => {
                // This node was the tail
                self.tail = node.prev;
            }
        }

        self.free_indices.push(idx);
        self.len -= 1;

        (node.key, node.value)
    }

    /// Remove and return the **tail** node (least-recently-used).
    ///
    /// This is the eviction path: when the cache is full, the LRU entry at the
    /// tail gets evicted to make room.
    pub fn pop_back(&mut self) -> Option<(K, V)> {
        self.tail.map(|idx| self.remove(idx))
    }

    /// Get a shared reference to the value at `idx`.
    pub fn get(&self, idx: usize) -> &V {
        &self.slots[idx]
            .as_ref()
            .expect("BUG: accessing empty slot")
            .value
    }

    /// Get a mutable reference to the value at `idx`.
    ///
    /// Used by `Shard::insert` to update the value of an existing key in-place
    /// (via `std::mem::replace`) without removing and re-inserting the node.
    pub fn get_mut(&mut self, idx: usize) -> &mut V {
        &mut self.slots[idx]
            .as_mut()
            .expect("BUG: accessing empty slot")
            .value
    }

    /// Number of live nodes in the list.
    pub fn len(&self) -> usize {
        self.len
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list() {
        let list: LruList<&str, i32> = LruList::with_capacity(4);
        assert_eq!(list.len(), 0);
        assert!(list.head.is_none());
        assert!(list.tail.is_none());
    }

    #[test]
    fn push_single_element() {
        let mut list = LruList::with_capacity(4);
        let idx = list.push_front("a", 1);
        assert_eq!(list.len(), 1);
        assert_eq!(*list.get(idx), 1);
        // Single element is both head and tail
        assert_eq!(list.head, Some(idx));
        assert_eq!(list.tail, Some(idx));
    }

    #[test]
    fn push_multiple_and_check_order() {
        let mut list = LruList::with_capacity(4);
        let a = list.push_front("a", 1);
        let _b = list.push_front("b", 2);
        let c = list.push_front("c", 3);

        // Head should be the last pushed (c), tail should be first pushed (a)
        assert_eq!(list.head, Some(c));
        assert_eq!(list.tail, Some(a));
        assert_eq!(list.len(), 3);

        // Pop from back should return items in push order (FIFO from tail)
        let (key, val) = list.pop_back().unwrap();
        assert_eq!((key, val), ("a", 1));

        let (key, val) = list.pop_back().unwrap();
        assert_eq!((key, val), ("b", 2));

        let (key, val) = list.pop_back().unwrap();
        assert_eq!((key, val), ("c", 3));

        assert_eq!(list.len(), 0);
        assert!(list.pop_back().is_none());
    }

    #[test]
    fn move_to_front_promotes_tail() {
        let mut list = LruList::with_capacity(4);
        let a = list.push_front("a", 1); // tail
        let _b = list.push_front("b", 2);
        let _c = list.push_front("c", 3); // head

        // Promote tail (a) to front
        list.move_to_front(a);
        assert_eq!(list.head, Some(a));

        // Now pop_back should return "b" (the new tail)
        let (key, _) = list.pop_back().unwrap();
        assert_eq!(key, "b");
    }

    #[test]
    fn move_to_front_head_is_noop() {
        let mut list = LruList::with_capacity(4);
        let _a = list.push_front("a", 1);
        let b = list.push_front("b", 2); // head

        list.move_to_front(b); // should be a no-op
        assert_eq!(list.head, Some(b));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn move_to_front_middle_element() {
        let mut list = LruList::with_capacity(4);
        let _a = list.push_front("a", 1); // tail
        let b = list.push_front("b", 2); // middle
        let _c = list.push_front("c", 3); // head

        list.move_to_front(b);
        assert_eq!(list.head, Some(b));

        // Order should now be: b → c → a
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "a");
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "c");
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "b");
    }

    #[test]
    fn remove_middle_element() {
        let mut list = LruList::with_capacity(4);
        let _a = list.push_front("a", 1);
        let b = list.push_front("b", 2);
        let _c = list.push_front("c", 3);

        let (key, val) = list.remove(b);
        assert_eq!((key, val), ("b", 2));
        assert_eq!(list.len(), 2);

        // Remaining: c → a
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "a");
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "c");
    }

    #[test]
    fn remove_head() {
        let mut list = LruList::with_capacity(4);
        let _a = list.push_front("a", 1);
        let b = list.push_front("b", 2);

        let (key, _) = list.remove(b);
        assert_eq!(key, "b");
        assert_eq!(list.len(), 1);
        assert_eq!(list.head, list.tail); // only one node left
    }

    #[test]
    fn free_list_reuse() {
        let mut list = LruList::with_capacity(4);
        let a = list.push_front("a", 1);
        list.remove(a); // frees slot 0

        let b = list.push_front("b", 2);
        // Should reuse slot 0
        assert_eq!(b, 0);
        assert_eq!(*list.get(b), 2);
    }

    #[test]
    fn get_mut_updates_value() {
        let mut list = LruList::with_capacity(4);
        let a = list.push_front("a", 1);
        *list.get_mut(a) = 42;
        assert_eq!(*list.get(a), 42);
    }

    #[test]
    fn single_element_pop_back() {
        let mut list = LruList::with_capacity(1);
        list.push_front("only", 99);

        let (k, v) = list.pop_back().unwrap();
        assert_eq!((k, v), ("only", 99));
        assert_eq!(list.len(), 0);
        assert!(list.head.is_none());
        assert!(list.tail.is_none());
    }

    #[test]
    fn two_element_move_to_front() {
        let mut list = LruList::with_capacity(2);
        let a = list.push_front("a", 1); // tail
        let _b = list.push_front("b", 2); // head

        // Promote a to front — now a is head, b is tail
        list.move_to_front(a);
        let (k, _) = list.pop_back().unwrap();
        assert_eq!(k, "b");
    }
}
