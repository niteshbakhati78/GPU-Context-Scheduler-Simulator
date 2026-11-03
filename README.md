# GPU Context Scheduler Simulator

A cycle-based C++ simulator in which several processes compete for a single GPU.
It models bounded command queues with back-pressure, three scheduling policies,
a configurable context-switch cost, and two preemption modes — then measures how
those choices trade off throughput, latency, fairness, and wasted cycles.

## Model

Time advances one cycle at a time. On each cycle the GPU does exactly one of:

- **work** — advance one unit of the current process's head job,
- **switch** — burn a cycle loading a different process (no work runs), or
- **idle** — nothing is runnable.

So `busyCycles + switchCycles + idleCycles == totalCycles` always holds (checked
by the test suite).

- **Processes** ([`Process`](src/simulator.h)) each have a priority, a workload
  (a shuffled mix of long compute jobs and short, latency-sensitive jobs), and a
  **bounded command queue**. A process is a *producer*: each cycle it tries to
  push its next job into the queue.
- **Back-pressure** — if the queue is full when a process wants to submit, the
  process **stalls** and we count the stalled cycle. This is the consumer (GPU)
  applying pressure back on the producer. Stall cycles are reported per process.
- **Jobs** ([`Job`](src/simulator.h)) carry `remaining` work. Preemption just
  stops decrementing it and switches away; the value persists, so a job resumes
  with its progress intact.

### Scheduling policies ([`select*` in src/simulator.h](src/simulator.h))

| Policy | Behaviour |
| --- | --- |
| **FIFO** | Non-preemptive. The oldest-enqueued head job runs to completion before another is picked. |
| **Round-robin** | Each process gets up to `timeslice` work cycles, then the GPU rotates to the next non-empty queue. Empty queues are skipped (so no switch is wasted on an idle process). |
| **Priority + preemption** | The highest-priority non-empty process runs. A higher-priority arrival either preempts immediately or waits — see below. |

### Context switching and preemption modes

- `switchCost` cycles are charged **whenever the loaded process changes**, for
  any policy.
- **Preemption modes** (priority policy):
  - **Immediate** — a higher-priority arrival preempts the running job *now* and
    pays an extra `saveCost` cycles to checkpoint GPU state.
  - **Wait-for-job** — the running job finishes first; no extra save cost.

## Metrics (per policy)

- **Throughput** — completed jobs per 1000 cycles (plus GPU utilization %).
- **Short-job wait** — average and **p99** of `finish - enqueue - serviceTime`
  over short (latency-sensitive) jobs. (Nearest-rank p99.)
- **% cycles lost to context switching** — `switchCycles / totalCycles`.
- **Fairness** — Jain's index over per-process work received:
  `J = (Σxᵢ)² / (n · Σxᵢ²)`, where `1.0` is perfectly fair.
- **Stall cycles per process** — cycles each process spent blocked on a full queue.

## Build & run

Requires a C++14 compiler. Tested with MinGW g++ 6.3.0 on Windows.

```sh
# experiments driver
g++ -std=c++14 -O2 -Wall -Wextra -o sim.exe   src/main.cpp
./sim.exe

# directed tests (exit code 0 == all passed)
g++ -std=c++14 -O2 -Wall -Wextra -o tests.exe src/tests.cpp
./tests.exe
```

## Experiments & findings

Numbers below are from the committed workload (`seed=12345`); rerun `sim.exe` to
reproduce. The point of each sweep is the *shape* of the trade-off, not the
absolute figures.

### 1. Timeslice length (round-robin)

| timeslice | thrpt (jobs/kc) | switch % | short p99 | util % |
| --- | --- | --- | --- | --- |
| 1 | 6.8 | 83.2 | 7523 | 16.8 |
| 4 | 18.0 | 55.3 | 2817 | 44.7 |
| 8 | 24.9 | 38.4 | 2059 | 61.6 |
| 16 | 30.7 | 23.9 | 1638 | 76.1 |
| 32 | 34.8 | 13.8 | 1464 | 86.2 |
| 64 | 37.3 | 7.5 | 1467 | 92.5 |
| 128 | 38.7 | 4.0 | 1348 | 96.0 |

Short slices rotate often, so response time improves — but with a fixed
`switchCost=5`, tiny slices spend most of the machine switching (83% lost at
`ts=1`!). As the slice grows, switch overhead falls and throughput climbs. The
**knee is around `ts=16–32`**: past there, throughput gains flatten (34.8 → 38.7
from 32 to 128) while p99 barely improves, so you are paying longer worst-case
waits for very little throughput. That is the sweet spot where the trade turns.

### 2. Preemption mode (priority)

Workload: one high-priority process emitting bursty short jobs against two
long-job processes, so every short arrival lands mid-long-job.

| mode | short avg | short p99 | thrpt (jobs/kc) | switch % |
| --- | --- | --- | --- | --- |
| immediate (save=5) | 9.8 | 10 | 23.44 | 15.0 |
| immediate (save=15) | 13.4 | 24 | 23.36 | 15.3 |
| immediate (save=40) | 24.2 | 45 | 22.98 | 16.7 |
| wait-for-job (save=0) | 32.7 | 71 | **26.46** | 4.1 |

Immediate preemption keeps short-job p99 tiny (10 cycles) because a short never
waits behind a long job — but every preemption costs a switch **plus** the save
cost, dragging throughput down and switch% up, and the penalty grows with
`saveCost`. Wait-for-job does the opposite: the fewest switches and the best
throughput (26.5), but shorts queue behind whole long jobs, pushing p99 to 71.
This is the classic latency-vs-throughput knob, and the save cost sets how
expensive the "low latency" side is.

### 3. Queue depth (round-robin)

| queueCap | total stall | thrpt (jobs/kc) | util % |
| --- | --- | --- | --- |
| 1 | 14067 | 24.78 | 61.4 |
| 2 | 13368 | 24.87 | 61.6 |
| 8 | 10746 | 24.87 | 61.6 |
| 16 | 7489 | 24.87 | 61.6 |
| 32 | 170 | 24.87 | 61.6 |
| 64 | 0 | 24.87 | 61.6 |

Deeper queues absorb producer bursts, so stalls fall steeply (14067 → 0). But
throughput is flat the whole way: the GPU is the bottleneck here, not the
buffers. Extra depth stops helping once the queue is big enough to hold the
jobs a process generates while waiting for its next turn — beyond that you are
just buying idle buffer space (and, in a real system, latency). The useful depth
is "just past the point where stalls hit zero," around `cap=32–64` for this
workload.

## Test plan

`src/tests.cpp` builds small, hand-crafted workloads with known-correct outcomes
and asserts on the resulting metrics / job timings. All checks pass.

| # | Directed check | How it is verified |
| --- | --- | --- |
| 1 | A higher-priority job preempts a running lower-priority one | A low-prio 100-cycle job starts at cycle 0; a high-prio short arrives at cycle 20. Assert the short *started while the long was still unfinished* and finished first. |
| 2 | No process starves under round-robin | Four processes with differing priorities; assert every process completes **all** its jobs (RR ignores priority, so none is locked out). |
| 3 | A preempted job resumes with progress intact | A long job is preempted repeatedly by bursty high-prio shorts. Assert work conservation (`busyCycles == totalWorkCompleted == 69`) and that each job completes exactly once — no lost or double-counted progress. |
| 4 | A full queue stalls without blocking others | A flooding process with `queueCap=2` stalls (back-pressure); assert the other process still completes all its jobs, and the stalled one eventually drains too. |
| 5 | Total work completed matches total work submitted | For all three policies: assert `totalWorkCompleted == Σ job.length` submitted, `busyCycles` equals that, and `switch + busy + idle == total`. |

## Files

- [`src/simulator.h`](src/simulator.h) — core model: jobs, processes, bounded
  queues, back-pressure, the three schedulers, preemption, and metrics.
- [`src/main.cpp`](src/main.cpp) — baseline policy comparison plus the three
  sweeps above.
- [`src/tests.cpp`](src/tests.cpp) — the directed test suite.
