// GPU Context Scheduler Simulator -- core model.
//
// One GPU, many processes. The GPU advances one cycle at a time. On each cycle
// it either (a) runs one unit of work for the currently-loaded process, (b) burns
// a cycle on a context switch, or (c) sits idle because nothing is runnable.
//
// Processes are producers: they push jobs into a bounded command queue. When the
// queue is full the process stalls (back-pressure) and we count those stalled
// cycles. The scheduler is the consumer, picking which process's queue head to
// run according to the active policy.
//
// C++14, header-only so the driver and the test suite share one definition.

#ifndef GPU_SCHEDULER_SIMULATOR_H
#define GPU_SCHEDULER_SIMULATOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace gpu {

enum class Policy { FIFO, RoundRobin, Priority };

// Only meaningful for the Priority policy. IMMEDIATE lets a higher-priority job
// preempt a running one mid-flight, paying `saveCost` extra cycles to checkpoint
// GPU state. WAIT lets the running job finish first, paying nothing extra.
enum class Preempt { Immediate, WaitForJob };

struct Job {
    int id = 0;
    int procId = 0;
    bool isShort = false; // short == latency-sensitive
    int length = 0;       // total work cycles required
    int remaining = 0;    // work cycles left (progress persists across preemption)
    long enqueueTime = -1;
    long startTime = -1;  // cycle the job first received the GPU
    long finishTime = -1;

    bool started() const { return startTime >= 0; }
};

struct Process {
    int id = 0;
    int priority = 0;      // larger == higher priority
    long arrivalGap = 0;   // min cycles between submissions (0 == greedy producer)
    long startDelay = 0;   // cycle before which this process submits nothing

    std::vector<Job> pending; // jobs this process will submit, in submission order
    std::deque<Job> queue;    // bounded command queue (the GPU's view)

    // runtime bookkeeping
    size_t nextSubmit = 0;
    long lastSubmit = -1;
    long stallCycles = 0;      // cycles spent wanting to submit but blocked by a full queue
    long completedWork = 0;    // useful work cycles this process received
    int completedJobs = 0;

    bool fullySubmitted() const { return nextSubmit >= pending.size(); }
    bool done() const { return fullySubmitted() && queue.empty(); }
};

struct Config {
    Policy policy = Policy::FIFO;
    Preempt preempt = Preempt::Immediate;
    int timeslice = 8;      // round-robin only
    int switchCost = 5;     // cycles lost per context switch (no work runs)
    int saveCost = 0;       // extra cycles when immediate-preempting a running job
    int queueCap = 8;       // bounded command queue depth
    long maxCycles = 5000000; // safety ceiling
};

struct Metrics {
    long totalCycles = 0;
    long busyCycles = 0;    // cycles that advanced real work
    long switchCycles = 0;  // cycles lost to context switching
    long idleCycles = 0;    // cycles with nothing runnable
    int completedJobs = 0;
    long totalWorkCompleted = 0; // sum of completed job lengths

    std::vector<long> perProcWork;
    std::vector<long> perProcStall;
    std::vector<int> perProcCompleted;

    std::vector<double> shortWaits; // wait = finish - enqueue - length, per short job
    std::vector<double> longWaits;
    std::vector<Job> finished;      // completed jobs with their timings (for inspection/tests)

    double throughputPerKiloCycle() const {
        return totalCycles ? 1000.0 * completedJobs / totalCycles : 0.0;
    }
    double switchPercent() const {
        return totalCycles ? 100.0 * switchCycles / totalCycles : 0.0;
    }
    double utilization() const {
        return totalCycles ? 100.0 * busyCycles / totalCycles : 0.0;
    }
    long totalStall() const {
        long s = 0;
        for (long v : perProcStall) s += v;
        return s;
    }
};

// ---- small statistics helpers -------------------------------------------------

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

// Nearest-rank p99.
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    long rank = (long)std::ceil(p / 100.0 * (double)v.size());
    if (rank < 1) rank = 1;
    if (rank > (long)v.size()) rank = (long)v.size();
    return v[(size_t)(rank - 1)];
}

// Jain's fairness index over per-process allocations. 1.0 == perfectly fair.
inline double jainIndex(const std::vector<long>& x) {
    if (x.empty()) return 0.0;
    double sum = 0.0, sumSq = 0.0;
    for (long v : x) { sum += (double)v; sumSq += (double)v * (double)v; }
    if (sumSq == 0.0) return 1.0; // nobody did any work -> treat as fair/degenerate
    return (sum * sum) / ((double)x.size() * sumSq);
}

// ---- simulator ----------------------------------------------------------------

class Simulator {
public:
    Simulator(std::vector<Process> procs, Config cfg)
        : procs_(std::move(procs)), cfg_(cfg) {}

    const Metrics& run() {
        const int n = (int)procs_.size();
        m_ = Metrics{};
        m_.perProcWork.assign(n, 0);
        m_.perProcStall.assign(n, 0);
        m_.perProcCompleted.assign(n, 0);

        long cycle = 0;
        for (; cycle < cfg_.maxCycles && !allDone(); ++cycle) {
            submitPhase(cycle);

            // (a) mid context switch: burn a cycle, no work.
            if (switchRemaining_ > 0) {
                --switchRemaining_;
                ++m_.switchCycles;
                if (switchRemaining_ == 0) current_ = pending_;
                continue;
            }

            int target = select();
            if (target < 0) { ++m_.idleCycles; continue; } // nothing runnable

            // (b) need to load a different process onto the GPU.
            if (target != current_) {
                bool preemptingRunningJob =
                    current_ >= 0 && !procs_[current_].queue.empty() &&
                    procs_[current_].queue.front().started() &&
                    procs_[current_].queue.front().remaining > 0;
                int cost = cfg_.switchCost;
                if (preemptingRunningJob && cfg_.policy == Policy::Priority &&
                    cfg_.preempt == Preempt::Immediate) {
                    cost += cfg_.saveCost;
                }
                pending_ = target;
                if (cost <= 0) {
                    current_ = target; // free switch -> fall through and run this cycle
                } else {
                    // Spend this cycle as the first switch cycle; the rest are
                    // consumed by the mid-switch branch on following cycles.
                    switchRemaining_ = cost - 1;
                    ++m_.switchCycles;
                    if (switchRemaining_ == 0) current_ = target;
                    continue;
                }
            }

            // (c) run one unit of work for the current process's head job.
            Process& p = procs_[current_];
            Job& job = p.queue.front();
            if (!job.started()) job.startTime = cycle;
            --job.remaining;
            --rrSliceRemaining_;
            ++m_.busyCycles;
            ++p.completedWork;
            ++m_.perProcWork[current_];

            if (job.remaining == 0) {
                job.finishTime = cycle + 1;
                recordCompletion(job);
                p.queue.pop_front();
                ++p.completedJobs;
                ++m_.perProcCompleted[current_];
            }
        }

        m_.totalCycles = cycle;
        for (int i = 0; i < n; ++i) m_.perProcStall[i] = procs_[i].stallCycles;
        return m_;
    }

    const std::vector<Process>& processes() const { return procs_; }

private:
    bool allDone() const {
        if (switchRemaining_ > 0) return false;
        for (const Process& p : procs_) if (!p.done()) return false;
        return true;
    }

    // Producers push jobs into their bounded queues; a process that is ready to
    // submit but finds its queue full stalls (back-pressure).
    void submitPhase(long cycle) {
        for (Process& p : procs_) {
            if (p.fullySubmitted()) continue;
            if (cycle < p.startDelay) continue;
            if (p.lastSubmit >= 0 && cycle - p.lastSubmit < p.arrivalGap) continue;
            if ((int)p.queue.size() >= cfg_.queueCap) {
                ++p.stallCycles; // wanted to submit, no room
                continue;
            }
            Job j = p.pending[p.nextSubmit++];
            j.enqueueTime = cycle;
            j.remaining = j.length;
            p.queue.push_back(j);
            p.lastSubmit = cycle;
        }
    }

    void recordCompletion(const Job& job) {
        ++m_.completedJobs;
        m_.totalWorkCompleted += job.length;
        m_.finished.push_back(job);
        double wait = (double)(job.finishTime - job.enqueueTime - job.length);
        if (wait < 0) wait = 0;
        if (job.isShort) m_.shortWaits.push_back(wait);
        else m_.longWaits.push_back(wait);
    }

    int select() {
        switch (cfg_.policy) {
            case Policy::FIFO: return selectFIFO();
            case Policy::RoundRobin: return selectRR();
            case Policy::Priority: return selectPriority();
        }
        return -1;
    }

    // Oldest enqueued head job wins; non-preemptive once a job has started.
    int selectFIFO() {
        if (runningUnfinished(current_)) return current_;
        int best = -1;
        long bestEnq = 0;
        for (int i = 0; i < (int)procs_.size(); ++i) {
            if (procs_[i].queue.empty()) continue;
            long enq = procs_[i].queue.front().enqueueTime;
            if (best < 0 || enq < bestEnq) { best = i; bestEnq = enq; }
        }
        return best;
    }

    // Each process gets up to `timeslice` work cycles, then we rotate.
    int selectRR() {
        if (current_ >= 0 && !procs_[current_].queue.empty() && rrSliceRemaining_ > 0)
            return current_;
        const int n = (int)procs_.size();
        int start = (current_ < 0) ? -1 : current_;
        for (int k = 1; k <= n; ++k) {
            int idx = ((start < 0 ? 0 : start) + k) % n;
            if (!procs_[idx].queue.empty()) { rrSliceRemaining_ = cfg_.timeslice; return idx; }
        }
        if (current_ >= 0 && !procs_[current_].queue.empty()) { // only runnable one
            rrSliceRemaining_ = cfg_.timeslice;
            return current_;
        }
        return -1;
    }

    // Highest-priority non-empty process. Immediate mode preempts a running
    // lower-priority job; wait mode lets the running job finish first.
    int selectPriority() {
        int best = -1, bestPrio = 0;
        for (int i = 0; i < (int)procs_.size(); ++i) {
            if (procs_[i].queue.empty()) continue;
            int prio = procs_[i].priority;
            if (best < 0 || prio > bestPrio) { best = i; bestPrio = prio; }
        }
        if (best < 0) return -1;
        if (runningUnfinished(current_)) {
            if (cfg_.preempt == Preempt::WaitForJob) return current_;
            if (procs_[best].priority > procs_[current_].priority) return best;
            return current_; // nothing strictly higher -> keep running
        }
        return best;
    }

    bool runningUnfinished(int idx) const {
        if (idx < 0 || procs_[idx].queue.empty()) return false;
        const Job& j = procs_[idx].queue.front();
        return j.started() && j.remaining > 0;
    }

    std::vector<Process> procs_;
    Config cfg_;
    Metrics m_;
    int current_ = -1;       // process currently loaded on the GPU
    int pending_ = -1;       // process being switched to
    long switchRemaining_ = 0;
    long rrSliceRemaining_ = 0;
};

// ---- workload construction ----------------------------------------------------

struct ProcSpec {
    int priority = 0;
    int numShort = 0;
    int numLong = 0;
    int shortLen = 4;
    int longLen = 40;
    int lenJitter = 0;   // +/- range on job length
    long arrivalGap = 0; // min cycles between submissions
};

// Deterministic workload from a seed: jobs per process are a shuffled mix of
// short and long jobs.
inline std::vector<Process> buildWorkload(const std::vector<ProcSpec>& specs, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Process> procs;
    int jobId = 0;
    for (int i = 0; i < (int)specs.size(); ++i) {
        const ProcSpec& s = specs[i];
        Process p;
        p.id = i;
        p.priority = s.priority;
        p.arrivalGap = s.arrivalGap;
        std::vector<Job> jobs;
        auto makeJob = [&](bool isShort, int base) {
            Job j;
            j.id = jobId++;
            j.procId = i;
            j.isShort = isShort;
            int len = base;
            if (s.lenJitter > 0) {
                std::uniform_int_distribution<int> d(-s.lenJitter, s.lenJitter);
                len = std::max(1, base + d(rng));
            }
            j.length = len;
            j.remaining = len;
            jobs.push_back(j);
        };
        for (int k = 0; k < s.numShort; ++k) makeJob(true, s.shortLen);
        for (int k = 0; k < s.numLong; ++k) makeJob(false, s.longLen);
        std::shuffle(jobs.begin(), jobs.end(), rng);
        p.pending = std::move(jobs);
        procs.push_back(std::move(p));
    }
    return procs;
}

} // namespace gpu

#endif // GPU_SCHEDULER_SIMULATOR_H
