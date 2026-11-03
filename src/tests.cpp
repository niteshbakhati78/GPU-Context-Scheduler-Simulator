// GPU Context Scheduler Simulator -- directed test suite.
//
// Each test builds a small, hand-crafted workload so the expected behaviour is
// unambiguous, then asserts on the resulting metrics / job timings.

#include "simulator.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace gpu;

static int g_failures = 0;
static std::string g_current;

static void check(bool cond, const std::string& what) {
    if (cond) {
        std::printf("    [PASS] %s\n", what.c_str());
    } else {
        std::printf("    [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

static void begin(const char* name) {
    g_current = name;
    std::printf("TEST: %s\n", name);
}

// Build a process holding explicit jobs (bypasses the random generator).
static Process makeProc(int id, int priority, const std::vector<Job>& jobs,
                        long startDelay = 0, long arrivalGap = 0) {
    Process p;
    p.id = id;
    p.priority = priority;
    p.startDelay = startDelay;
    p.arrivalGap = arrivalGap;
    for (Job j : jobs) {
        j.procId = id;
        j.remaining = j.length;
        p.pending.push_back(j);
    }
    return p;
}

static Job job(int id, bool isShort, int length) {
    Job j; j.id = id; j.isShort = isShort; j.length = length; j.remaining = length; return j;
}

static const Job* findFinished(const Metrics& m, int jobId) {
    for (const Job& j : m.finished) if (j.id == jobId) return &j;
    return nullptr;
}

// 1. A higher-priority job preempts a lower-priority one that is mid-flight.
static void testPreemption() {
    begin("higher-priority job preempts a running lower-priority one");
    std::vector<Process> procs;
    procs.push_back(makeProc(0, /*prio*/1, {job(/*id*/100, false, /*len*/100)}));        // low, starts at 0
    procs.push_back(makeProc(1, /*prio*/5, {job(/*id*/200, true, /*len*/5)}, /*delay*/20)); // high, arrives at 20

    Config c; c.policy = Policy::Priority; c.preempt = Preempt::Immediate;
    c.switchCost = 2; c.saveCost = 3; c.queueCap = 8;
    Simulator sim(procs, c);
    const Metrics& m = sim.run();

    const Job* low = findFinished(m, 100);
    const Job* high = findFinished(m, 200);
    check(low && high, "both jobs completed");
    if (low && high) {
        check(low->startTime >= 0 && low->startTime < 20, "low-priority job started before high arrived");
        check(high->finishTime < low->finishTime, "high-priority job finished before the preempted low one");
        check(high->startTime >= 20 && high->startTime < low->finishTime,
              "high-priority job ran while the low one was still unfinished (preemption)");
    }
}

// 2. No process starves under round-robin: everyone's jobs complete.
static void testNoStarvationRR() {
    begin("no process starves under round-robin");
    std::vector<Process> procs;
    for (int i = 0; i < 4; ++i) {
        std::vector<Job> jobs;
        for (int k = 0; k < 5; ++k) jobs.push_back(job(i * 100 + k, k % 2 == 0, 10 + i * 3));
        procs.push_back(makeProc(i, /*prio*/i, jobs)); // priorities vary but RR ignores them
    }
    Config c; c.policy = Policy::RoundRobin; c.timeslice = 4; c.switchCost = 2; c.queueCap = 4;
    Simulator sim(procs, c);
    sim.run();
    bool allFinished = true;
    for (const Process& p : sim.processes())
        if (p.completedJobs != (int)p.pending.size()) allFinished = false;
    check(allFinished, "every process completed all of its jobs (none starved)");
}

// 3. A preempted job resumes with its progress intact (no lost or duplicated work).
static void testProgressIntact() {
    begin("a preempted job resumes with progress intact");
    std::vector<Process> procs;
    procs.push_back(makeProc(0, 1, {job(100, false, 60)}));                      // long, preempted repeatedly
    procs.push_back(makeProc(1, 9, {job(200, true, 3), job(201, true, 3),
                                    job(202, true, 3)}, /*delay*/10, /*gap*/15)); // bursty high-prio shorts
    Config c; c.policy = Policy::Priority; c.preempt = Preempt::Immediate;
    c.switchCost = 2; c.saveCost = 4; c.queueCap = 8;
    Simulator sim(procs, c);
    const Metrics& m = sim.run();

    const Job* low = findFinished(m, 100);
    check(low != nullptr, "the repeatedly-preempted long job eventually completed");
    // Work conservation proves no progress was lost or double-counted.
    check(m.busyCycles == m.totalWorkCompleted,
          "work cycles spent == total work completed (60+3+3+3 == " +
          std::to_string(m.totalWorkCompleted) + ")");
    check(m.totalWorkCompleted == 69, "exactly the submitted work (69 cycles) was executed");
    check(m.completedJobs == 4, "all four jobs completed exactly once");
}

// 4. A process with a full queue stalls without blocking the others.
static void testBackpressureIsolation() {
    begin("a process with a full queue stalls without blocking others");
    std::vector<Process> procs;
    // P0: floods the GPU with long work (greedy) -> its queue fills and it stalls.
    std::vector<Job> flood;
    for (int k = 0; k < 20; ++k) flood.push_back(job(k, false, 40));
    procs.push_back(makeProc(0, /*prio*/1, flood));
    // P1: a handful of short jobs that must still get through.
    std::vector<Job> light;
    for (int k = 0; k < 6; ++k) light.push_back(job(1000 + k, true, 4));
    procs.push_back(makeProc(1, /*prio*/1, light, /*delay*/0, /*gap*/5));

    Config c; c.policy = Policy::RoundRobin; c.timeslice = 6; c.switchCost = 2; c.queueCap = 2;
    Simulator sim(procs, c);
    sim.run();
    const Process& p0 = sim.processes()[0];
    const Process& p1 = sim.processes()[1];
    check(p0.stallCycles > 0, "flooding process stalled on its full queue (back-pressure engaged)");
    check(p1.completedJobs == (int)p1.pending.size(),
          "the other process still completed all its jobs despite the stall");
    check(p0.completedJobs == (int)p0.pending.size(),
          "the stalled process also eventually drained and finished");
}

// 5. Total work completed matches total work submitted.
static void testWorkConservation() {
    begin("total work completed matches total work submitted");
    auto specs = std::vector<ProcSpec>{
        {3, 30, 3, 4, 30, 2, 4},
        {2, 15, 8, 5, 35, 4, 1},
        {1, 5, 20, 6, 50, 8, 0},
    };
    long submitted = 0;
    for (const Process& p : buildWorkload(specs, 777))
        for (const Job& j : p.pending) submitted += j.length;

    const Policy policies[] = {Policy::FIFO, Policy::RoundRobin, Policy::Priority};
    for (Policy pol : policies) {
        Config c; c.policy = pol; c.timeslice = 8; c.switchCost = 5; c.saveCost = 10; c.queueCap = 8;
        Simulator sim(buildWorkload(specs, 777), c);
        const Metrics& m = sim.run();
        std::string n = (pol == Policy::FIFO) ? "FIFO" :
                        (pol == Policy::RoundRobin) ? "RoundRobin" : "Priority";
        check(m.totalWorkCompleted == submitted,
              n + ": completed work (" + std::to_string(m.totalWorkCompleted) +
              ") == submitted work (" + std::to_string(submitted) + ")");
        check(m.busyCycles == submitted, n + ": busy cycles account for exactly the submitted work");
        check(m.switchCycles + m.busyCycles + m.idleCycles == m.totalCycles,
              n + ": cycle accounting is complete (switch + busy + idle == total)");
    }
}

int main() {
    std::printf("GPU Context Scheduler Simulator -- directed tests\n\n");
    testPreemption();
    testNoStarvationRR();
    testProgressIntact();
    testBackpressureIsolation();
    testWorkConservation();
    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
