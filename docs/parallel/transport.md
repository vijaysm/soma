# Record transport

The transport layer (`soma/parallel/transport/`, namespace `soma::transport`) moves fixed-width
records between ranks when no rank knows in advance who will send it what. SOMA's distributed
algorithms (sharing, ghosts, migration, matching, derived gids, halo exchange setup) are built on
it. You need it when writing a distributed algorithm of your own over SOMA meshes; for moving
field values between owners and copies, a [halo exchange](ghosts-and-halos.md) is the simpler
tool.

| Header | Provides |
|---|---|
| `soma/parallel/transport/route.hpp` | `RecordBatch`, `route`, `Strategy`: sparse personalised exchange. |
| `soma/parallel/transport/comm_plan.hpp` | `CommPlan`, `learn_plan`: replaying an exchange's pattern with point-to-point messages. |
| `soma/parallel/transport/rendezvous.hpp` | `owner`, `rendezvous`, `rendezvous_to`: a distributed directory keyed by 64-bit keys. |
| `soma/parallel/transport/sort.hpp` | `rank_keys`, `rank_keys_held`: global ranks of keys by a sample sort. |

## Records and routing

```cpp
template <class MemSpace>
struct RecordBatch {
  Kokkos::View<int*, MemSpace> dest;    // the rank each record goes to
  Kokkos::View<int*, MemSpace> source;  // the rank it came from, filled by route()
  Kokkos::View<std::byte**, Kokkos::LayoutRight, MemSpace> payload;  // one row per record

  RecordBatch(std::size_t n, std::size_t width, const std::string& label = "transport::RecordBatch");
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t width() const;
};

enum class Strategy : std::uint8_t { automatic, crystal, alltoallv, nbx };

template <class ExecSpace>
RecordBatch<typename ExecSpace::memory_space> route(const ExecSpace&, const Comm& comm,
                                                    const RecordBatch<typename ExecSpace::memory_space>& in,
                                                    Strategy strategy = Strategy::automatic);
```

Each rank fills a batch of records, each naming the rank it goes to, with a payload row of
`width` bytes; the width must be the same on every rank (it may be 0, so a record carries only
its source). `route` is collective and returns the records of every rank's batch bound for this
rank, with `dest` set to this rank and `source` to the sender's.

**Order.** Arrived records are grouped by source rank, ascending, and within a source in the
sender's order. A rank's records to itself do not travel but appear in their place among the
sources.

Adapted from the `HaloExchange` constructor (`soma/parallel/halo_exchange.hpp`), where each rank asks the
owner of each of its ghosts by gid:

```cpp
using Batch = transport::RecordBatch<Kokkos::HostSpace>;
Batch asks(n_ghosts, sizeof(GO), "soma::HaloExchange::ghosts");
for (std::size_t i = 0; i < n_ghosts; ++i) {
  asks.dest(i) = owner_of_ghost(i);
  std::memcpy(&asks.payload(i, 0), &gid_of_ghost(i), sizeof(GO));
}
const Batch asked = transport::route(Kokkos::DefaultHostExecutionSpace{}, comm, asks);
// asked.source(i) holds ghost gid asked.payload(i, ...)
```

**Errors.** Thrown on every rank: `std::invalid_argument` if any rank names a destination outside
the communicator, passes a payload that is not one row of `width` bytes per record, or differs
from another rank in width or strategy. `std::runtime_error` if a message is inconsistent with
what the ranks sent; incoming sizes are checked before any buffer is sized from them.

**Memory.** Records are staged through host memory, one copy in and one out, whatever the memory
space of the batch.

**Tags.** Messages use tags 32000 to 32767 of the communicator.

### Strategies

All strategies return the same records in the same order; they differ in messages and memory.

| Strategy | How | Messages per rank |
|---|---|---|
| `crystal` | The crystal router (Tufo and Fischer): the active ranks are halved $\lceil \log_2 P \rceil$ times, each rank sending the records bound for the other half to one partner. No arrays of size $P$; records travel up to $\log_2 P$ hops. | $O(\log P)$ |
| `alltoallv` | Counts by `alltoall`, then one `MPI_Alltoallv`. | collective |
| `nbx` | A synchronous send per destination, then a non-blocking barrier once they are received (Hoefler, Siebert and Lumsdaine, 2010). No forwarding and no counts of size $P$. | one per destination |
| `automatic` (default) | `alltoallv` up to 64 ranks, `crystal` above. | |

The 64-rank threshold of `automatic` is provisional, not a tuned value; pass a strategy
explicitly when you have measured one to be better on your machine.

## Communication plans

When the same pattern of records is exchanged repeatedly, routing each time is wasteful. A
`CommPlan` records who sends how many records to whom and replays the exchange with one round of
point-to-point messages to the known neighbours.

```cpp
template <class MemSpace>
CommPlan learn_plan(const Comm& comm, const RecordBatch<MemSpace>& sent, const RecordBatch<MemSpace>& arrived);
```

A plan is learnt from both halves of one routed exchange, the destinations this rank sent and
the sources it received, so no handshake is needed. From the header:

```cpp
auto arrived = transport::route(exec, comm, sent);
const auto plan = transport::learn_plan(comm, sent, arrived);
plan.forward(comm, values_per_sent_record, values_per_arrived_record);
plan.reverse(comm, answers_per_arrived_record, answers_per_sent_record);
```

- `forward(comm, sent, arrived)` moves one value per sent record to the rank it was routed to, in
  `route`'s order there.
- `reverse(comm, arrived, sent)` sends one value per arrived record back to its place among the
  sender's records, for answering requests.
- Values are `std::span`s of any trivially copyable `T` (or bytes, with `forward_bytes` and
  `reverse_bytes`). A span without one value per record throws `std::invalid_argument` before any
  message is sent.
- A rank's records to itself are copied, not sent.
- `forward` and `reverse` exchange messages with the plan's neighbours only (`send_ranks()`,
  `recv_ranks()`); call them on every rank, in the same order. They use tags 32600 and 32601.
- `num_sent()` and `num_arrived()` give the record counts.

`CommPlan::from_counts(comm, sent, arrived)` makes a plan from per-rank counts (`Run{rank,
count}`) when both sides already know them and hold their records grouped by rank, ranks strictly
ascending; each neighbour's run then leaves as one message without being copied. It throws
`std::invalid_argument` for a rank outside the communicator, ranks not strictly ascending, or a
count to this rank that differs between sent and arrived. The constructor from destinations and
sources throws likewise.

### Overlapping exchanges

```cpp
template <class T>
[[nodiscard]] Pending begin_forward(const Comm& comm, std::span<const T> sent, std::span<T> arrived, int tag) const;
```

`begin_forward` posts the receives and sends on a tag of the caller's choosing, copies this
rank's own records, and returns a `CommPlan::Pending`; its `wait()` (or destructor) finishes the
exchange, and a second `wait()` does nothing. Several forwards on different tags may be in flight
at once, finished in any order, with work in between. The spans must outlive the `Pending`. From
`tests/unit/test_transport.cpp`:

```cpp
auto pa = plan.begin_forward(c, std::span<const std::int64_t>(sent), std::span<std::int64_t>(a), 7101);
auto pb = plan.begin_forward(c, std::span<const std::int64_t>(twice), std::span<std::int64_t>(b), 7102);
pb.wait();
pa.wait();
```

## Rendezvous: a distributed directory

```cpp
[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t z) noexcept;     // splitmix64 finaliser
[[nodiscard]] constexpr int owner(std::uint64_t key, int size) noexcept;  // mix(key) % size

template <class ExecSpace, class Answer>
Kokkos::View<std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space> rendezvous(
    const ExecSpace& exec, const Comm& comm,
    const Kokkos::View<const std::uint64_t*, typename ExecSpace::memory_space>& keys,
    const Kokkos::View<const std::byte**, Kokkos::LayoutRight, typename ExecSpace::memory_space>& payload,
    std::size_t answer_width, const Answer& answer, Strategy strategy = Strategy::automatic);
```

Each key, with its payload row, goes to the key's owner, `owner(key, P)`, which depends only on
the key and $P$. The owner sees every record sent for its keys and answers each. The callback
`answer(keys, arrived)` receives the keys that arrived on this rank (a
`Kokkos::View<std::uint64_t*>`) and a `RecordBatch` of their senders and payloads, grouped by
source rank in each source's order, and returns a `Kokkos::View<std::byte**, Kokkos::LayoutRight>`
of one `answer_width`-byte row per arrival. The answers come back to the senders in the order of
their keys. `resolve_sharing` uses the same placement, sending each gid to `owner(gid, P)`, to
find which ranks hold it ([distributed meshes](distributed-meshes.md)).

`rendezvous_to(..., answer, dest_of, strategy)` sends each key to `dest_of(key)`, a rank of
`comm`, instead: a directory placed by the caller (blocks of ids, say).

Both are collective. They throw `std::invalid_argument` on every rank if any rank passes a
payload without a row per key, names a rank outside `comm`, or returns answers that are not a row
of `answer_width` bytes per arrival. An exception from `answer` on one rank is rethrown there, and
`std::runtime_error` is thrown on the others.

## Global ranks of keys

```cpp
struct KeyRanks {
  std::vector<std::uint64_t> index;  // per key, in the order given
  std::uint64_t total = 0;           // the number of indices on all ranks
};

[[nodiscard]] KeyRanks rank_keys(const Comm& comm, std::span<const std::uint64_t> keys, std::size_t words,
                                 bool distinct, Strategy strategy = Strategy::automatic);

[[nodiscard]] HeldKeyRanks rank_keys_held(const Comm& comm, std::span<const std::uint64_t> keys, std::size_t words,
                                          Strategy strategy = Strategy::automatic);
```

A key is `words` consecutive `std::uint64_t`, compared lexicographically. `rank_keys` is a
collective sample sort: each rank sends regular samples of its sorted keys to rank 0, which picks
$P - 1$ splitters and broadcasts them; every key is routed to the rank of its key range, ranked
there, offset by an exclusive scan of the counts, and its index sent back.

| `distinct` | Indices |
|---|---|
| `true` | Equal keys, on any ranks, share one index; indices number the distinct keys in ascending order, so they do not depend on $P$. This is how gids of derived entities are numbered. |
| `false` | Every key has its own index; equal keys are ordered by sender rank, then the sender's order. |

`rank_keys_held` gives distinct indices and also, per key, the other ranks that gave an equal key
(`offsets`, `ranks`, as in `Sharing`), at the cost of one more exchange of the keys held on several
ranks only. Memory per rank is the keys it holds and receives plus $P - 1$ splitters.

Both throw `std::invalid_argument` on every rank if any rank passes `words == 0`, keys that are
not whole rows, or a different `words` or `distinct` than another rank.
