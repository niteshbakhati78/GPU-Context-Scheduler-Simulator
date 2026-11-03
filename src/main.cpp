// GPU Context Scheduler Simulator -- experiments driver.
//
// Runs a baseline comparison of the three policies, then the three sweeps that
// expose the real trade-offs: timeslice length, preemption mode, and queue depth.

#include "simulator.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace gpu;

// A representative mixed workload: four processes competing for one GPU.
//  - P0: high-priority, latency-sensitive, bursty short jobs.
//  - P1: medium-priority mix.
//  - P2/P3: low-priority, compute-heavy, greedy producers (create back-pressure).
static std::vector<ProcSpec> baselineSpecs() {
    std::vector<ProcSpec> s(4);
    s[0] = {/*prio*/3, /*short*/40, /*long*/2,  /*sLen*/4,  /*lLen*/30, /*jit*/2, /*gap*/6};
    s[1] = {/*prio*/2, /*short*/20, /*long*/10, /*sLen*/5,  /*lLen*/35, /*jit*/4, /*gap*/2};
    s[2] = {/*prio*/1, /*short*/4,  /*long*/25, /*sLen*/6,  /*lLen*/50, /*jit*/8, /*gap*/0};
    s[3] = {/*prio*/1, /*short*/4,  /*long*/25, /*sLen*/6,  /*lLen*/50, /*jit*/8, /*gap*/0};
    return s;
}

// Dedicated workload for the preemption experiment: ONE high-priority process
// emitting bursty, latency-sensitive short jobs, and two low-priority processes
// grinding long compute jobs. Every short job here is one we care about, so
// shortWaits measures exactly the latency-sensitive traffic -- and each short
// arrival lands while a long job is mid-flight, forcing a preempt-or-wait choice.
static std::vector<ProcSpec> preemptSpecs() {
    std::vector<ProcSpec> s(3);
    s[0] = {/*prio*/9, /*short*/60, /*long*/0,  /*sLen*/4,  /*lLen*/0,  /*jit*/1,  /*gap*/25};
    s[1] = {/*prio*/1, /*short*/0,  /*long*/40, /*sLen*/0,  /*lLen*/60, /*jit*/10, /*gap*/0};
    s[2] = {/*prio*/1, /*short*/0,  /*long*/40, /*sLen*/0,  /*lLen*/60, /*jit*/10, /*gap*/0};
    return s;
}

static const unsigned kSeed = 12345;

static Metrics runWith(Config cfg) {
    Simulator sim(buildWorkload(baselineSpecs(), kSeed), cfg);
    return sim.run();
}

static Metrics runWith(Config cfg, const std::vector<ProcSpec>& specs) {
    Simulator sim(buildWorkload(specs, kSeed), cfg);
    return sim.run();
}

static void printReport(const std::string& label, const Metrics& m) {
    std::printf("  %-22s thrpt=%6.3f jobs/kc  util=%5.1f%%  switch=%5.1f%%\n",
                label.c_str(), m.throughputPerKiloCycle(), m.utilization(), m.switchPercent());
    std::printf("  %-22s short-wait avg=%8.1f  p99=%8.1f (n=%lu)\n",
                "", mean(m.shortWaits), percentile(m.shortWaits, 99.0),
                (unsigned long)m.shortWaits.size());
    std::printf("  %-22s fairness(Jain)=%.3f  total-stall=%ld cycles\n",
                "", jainIndex(m.perProcWork), m.totalStall());
    std::printf("  %-22s per-proc stall:", "");
    for (size_t i = 0; i < m.perProcStall.size(); ++i)
        std::printf(" P%lu=%ld", (unsigned long)i, m.perProcStall[i]);
    std::printf("\n");
}

static void section(const char* title) {
    std::printf("\n==================================================================\n");
    std::printf("%s\n", title);
    std::printf("==================================================================\n");
}

static void comparePolicies() {
    section("BASELINE: policy comparison (switchCost=5, timeslice=8, queueCap=8)");
    {
        Config c; c.policy = Policy::FIFO; c.switchCost = 5; c.queueCap = 8;
        printReport("FIFO", runWith(c));
    }
    {
        Config c; c.policy = Policy::RoundRobin; c.timeslice = 8; c.switchCost = 5; c.queueCap = 8;
        printReport("RoundRobin(ts=8)", runWith(c));
    }
    {
        Config c; c.policy = Policy::Priority; c.preempt = Preempt::Immediate;
        c.saveCost = 15; c.switchCost = 5; c.queueCap = 8;
        printReport("Priority(immediate)", runWith(c));
    }
}

// Experiment 1: short timeslices improve response but lose cycles to switching.
static void sweepTimeslice() {
    section("EXPERIMENT 1: round-robin timeslice sweep (switchCost=5, queueCap=8)");
    std::printf("  %8s %14s %12s %14s %10s\n",
                "timeslice", "thrpt(jobs/kc)", "switch%", "short-p99", "util%");
    for (int ts : {1, 2, 4, 8, 16, 32, 64, 128}) {
        Config c; c.policy = Policy::RoundRobin; c.timeslice = ts; c.switchCost = 5; c.queueCap = 8;
        Metrics m = runWith(c);
        std::printf("  %8d %14.3f %11.1f%% %14.1f %9.1f%%\n",
                    ts, m.throughputPerKiloCycle(), m.switchPercent(),
                    percentile(m.shortWaits, 99.0), m.utilization());
    }
    std::printf("  -> short slices cut p99 wait but raise switch%%; the knee is where\n");
    std::printf("     throughput stops improving as the slice grows.\n");
}

// Experiment 2: immediate preemption (high save cost) vs wait-for-job.
static void comparePreemption() {
    section("EXPERIMENT 2: preemption mode -- p99 short wait vs throughput cost");
    std::printf("  (1 high-prio bursty short-job process vs 2 long-job processes)\n");
    std::printf("  %-26s %12s %12s %14s %10s\n",
                "mode", "short-avg", "short-p99", "thrpt(jobs/kc)", "switch%");
    for (int saveCost : {5, 15, 40}) {
        Config c; c.policy = Policy::Priority; c.preempt = Preempt::Immediate;
        c.saveCost = saveCost; c.switchCost = 5; c.queueCap = 8;
        Metrics m = runWith(c, preemptSpecs());
        std::string lbl = "immediate(save=" + std::to_string(saveCost) + ")";
        std::printf("  %-26s %12.1f %12.1f %14.3f %9.1f%%\n", lbl.c_str(),
                    mean(m.shortWaits), percentile(m.shortWaits, 99.0),
                    m.throughputPerKiloCycle(), m.switchPercent());
    }
    {
        Config c; c.policy = Policy::Priority; c.preempt = Preempt::WaitForJob;
        c.saveCost = 0; c.switchCost = 5; c.queueCap = 8;
        Metrics m = runWith(c, preemptSpecs());
        std::printf("  %-26s %12.1f %12.1f %14.3f %9.1f%%\n", "wait-for-job(save=0)",
                    mean(m.shortWaits), percentile(m.shortWaits, 99.0),
                    m.throughputPerKiloCycle(), m.switchPercent());
    }
    std::printf("  -> immediate preemption buys low short-job p99 at a throughput cost that\n");
    std::printf("     grows with save cost; wait-for-job is the opposite trade.\n");
}

// Experiment 3: deeper queues reduce stalls -- up to a point.
static void sweepQueueDepth() {
    section("EXPERIMENT 3: queue depth sweep (round-robin ts=8, switchCost=5)");
    std::printf("  %9s %14s %14s %12s\n", "queueCap", "total-stall", "thrpt(jobs/kc)", "util%");
    for (int cap : {1, 2, 4, 8, 16, 32, 64}) {
        Config c; c.policy = Policy::RoundRobin; c.timeslice = 8; c.switchCost = 5; c.queueCap = cap;
        Metrics m = runWith(c);
        std::printf("  %9d %14ld %14.3f %11.1f%%\n",
                    cap, m.totalStall(), m.throughputPerKiloCycle(), m.utilization());
    }
    std::printf("  -> stalls fall fast as depth grows, then flatten: beyond the point where\n");
    std::printf("     the GPU is the bottleneck, extra buffering stops helping.\n");
}

int main() {
    std::printf("GPU Context Scheduler Simulator\n");
    std::printf("Workload: 4 processes, seed=%u. Short=latency-sensitive, Long=compute.\n", kSeed);
    comparePolicies();
    sweepTimeslice();
    comparePreemption();
    sweepQueueDepth();
    std::printf("\n");
    return 0;
}
