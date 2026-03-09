import json
import os

# Shared long prefix (the same for all queries)
system_prompt = """You are an AI assistant specialized in operating systems.
The following document discusses the Multi-Level Feedback Queue (MLFQ) scheduler in depth.
Use this document to answer questions concisely and accurately.

1. Introduction
The Multi-Level Feedback Queue (MLFQ) scheduler is one of the most widely studied and implemented scheduling algorithms in operating systems. It was first described in 1962 by Corbató et al. in the Compatible Time-Sharing System (CTSS), work that later contributed to Corbató receiving the Turing Award. MLFQ has since evolved and influenced modern systems like Solaris, BSD, and Windows.
MLFQ addresses two key goals of CPU scheduling:
Minimize turnaround time: Favor short jobs so they finish quickly. Traditional algorithms like Shortest Job First (SJF) or Shortest Time-to-Completion First (STCF) achieve this but require knowledge of job lengths, which is generally unavailable.
Minimize response time: Ensure interactive jobs (those waiting for user input/output) get prompt service, making the system responsive. Round-Robin scheduling helps here but performs poorly for long-running CPU-bound jobs.
The core challenge is:
How can we design a scheduler that optimizes both turnaround and response time without knowing job lengths in advance?
MLFQ’s answer: learn from history. It observes job behavior at runtime (CPU bursts vs. I/O waits) and adjusts priorities accordingly. Jobs that behave like interactive processes retain high priority; jobs that are CPU-intensive gradually drop to lower priority.

2. MLFQ Basics
MLFQ maintains multiple queues, each representing a priority level. Higher queues have higher priority. At any point:
Rule 1: If Priority(A) > Priority(B), then A runs (B waits).
Rule 2: If Priority(A) = Priority(B), they run in Round Robin within that queue.
Thus, higher-priority queues dominate scheduling. The innovation lies in priority adjustment, which is not fixed but depends on observed job behavior.
If a process frequently relinquishes the CPU early (like an interactive job waiting for input), it stays at a high priority. If it hogs the CPU (like a batch job), it is demoted. This makes MLFQ adaptive and self-tuning.

3. First Attempt: Simple Priority Adjustment
To classify jobs, MLFQ introduces the concept of allotment: the maximum CPU time a job can spend at its current priority before being demoted. Initially, allotment equals one time slice.
Rule 3: When a job enters, it starts at the highest priority.
Rule 4a: If a job uses up its allotment, its priority is reduced (it moves down one queue).
Rule 4b: If a job voluntarily relinquishes the CPU before its allotment ends (e.g., due to I/O), it stays in the same queue.
Examples:
Long-running job: A single CPU-bound job quickly drops from high to low priority and stays there.
Short interactive job arriving later: When a short job appears, it enters at high priority, runs quickly, and completes before dropping down — thus approximating Shortest Job First.
Interactive job with frequent I/O: Since it repeatedly gives up the CPU early, it stays at high priority and receives quick responses.
This basic design balances responsiveness with fairness. However, flaws remain.

4. Problems with Basic MLFQ
Starvation: If many interactive jobs exist, CPU-bound jobs may starve forever at the lowest queue.
Gaming the scheduler: A clever process can repeatedly perform I/O just before its allotment expires to stay at high priority indefinitely, monopolizing CPU.
Changing job behavior: A process might start as CPU-bound but later become interactive. Without flexibility, it would be stuck at low priority.

5. Attempt #2: Priority Boost
To fix starvation and handle behavior changes, MLFQ introduces priority boosts:
Rule 5: After a fixed period S, all jobs are moved to the highest-priority queue.
This ensures:
Starving CPU-bound jobs eventually get CPU time.
Jobs that change behavior are reclassified appropriately.
Choosing S is tricky. If too large, starvation persists; if too small, interactive jobs lose responsiveness. Ousterhout referred to such constants as “voodoo constants”, since tuning them often feels arbitrary and workload-dependent. Today, some systems use automatic methods (even machine learning) to tune such parameters.

6. Attempt #3: Better Accounting
The remaining flaw is gaming. With Rules 4a/4b, a process could exploit I/O to retain priority unfairly.
Solution: Refine accounting.
Rule 4 (revised): Once a job consumes its total allotment at a given level, regardless of how it uses it (one burst or many short ones), its priority is reduced.
This prevents manipulation. Jobs inevitably move down if they are CPU-intensive, even if they try to trick the system with frequent I/O calls.

7. Tuning MLFQ and Practical Issues
In real systems, implementing MLFQ requires tuning multiple parameters:
Number of queues: More queues allow finer distinctions but add complexity.
Time-slice length per queue: Typically shorter for high-priority queues (e.g., 10ms) and longer for low-priority ones (e.g., 100ms). This makes sense: interactive jobs need responsiveness, CPU-bound jobs benefit from longer slices to reduce overhead.
"""

# Short user questions
queries = [
        "What is the main goal of MLFQ scheduling?",
    "Who first described the MLFQ scheduler and in what system?",
    "What are the two primary problems MLFQ tries to solve?",
    "Why can’t SJF or STCF be directly applied in practice?",
    "Why is Round Robin bad for turnaround time?",
    "What is the crux question behind MLFQ scheduling?",
    "How does MLFQ attempt to predict job behavior?",
    "What does 'learning from history' mean in MLFQ?",
    "Why does MLFQ use multiple queues?",
    "How are jobs assigned to queues in MLFQ?",
    "What is Rule 1 of MLFQ?",
    "What is Rule 2 of MLFQ?",
    "How does MLFQ handle jobs with equal priority?",
    "How does MLFQ decide which job runs first?",
    "Why do interactive jobs usually retain high priority?",
    "Why are CPU-intensive jobs demoted in MLFQ?",
    "What is a job’s allotment in MLFQ?",
    "What happens if a job uses up its allotment?",
    "What happens if a job relinquishes the CPU early?",
    "What happens to a single long-running job in MLFQ?",
    "How does MLFQ approximate SJF?",
    "How are short jobs favored in MLFQ?",
    "What happens when an interactive job frequently gives up the CPU?",
    "Why does MLFQ risk starvation of long-running jobs?",
    "What is gaming the scheduler in MLFQ?",
    "How can a job game the scheduler under old rules?",
    "What is the effect of priority boost in MLFQ?",
    "What problem does Rule 5 solve?",
    "How often should priority boosts occur?",
    "Why are such constants called 'voodoo constants'?",
    "What is Ousterhout’s Law?",
    "How can better accounting prevent gaming in MLFQ?",
    "What is the updated Rule 4 for MLFQ?",
    "Why does MLFQ need better accounting?",
    "What is starvation in MLFQ?",
    "How can MLFQ adapt to a job changing behavior over time?",
    "How many queues are typically used in MLFQ?",
    "Why are higher-priority queues given shorter time slices?",
    "Why are lower-priority queues given longer time slices?",
    "What is the Solaris MLFQ implementation called?",
    "How many queues does Solaris TS typically use?",
    "What does FreeBSD use instead of exact rules?",
    "What is decay-usage scheduling?",
    "Why might some queues be reserved for OS work?",
    "What does the 'nice' utility do in scheduling?",
    "What is the purpose of advice in scheduling?",
    "How does MLFQ handle both interactive and CPU-bound jobs?",
    "What is the fundamental advantage of MLFQ?",
    "Which real systems implement MLFQ variants?",
    "How does MLFQ balance fairness with performance?",
    "Why does MLFQ not require a priori job knowledge?",
    "What is turnaround time in scheduling?",
    "What is response time in scheduling?",
    "How does MLFQ optimize turnaround time?",
    "How does MLFQ minimize response time?",
    "What is the Compatible Time-Sharing System (CTSS)?",
    "What later system refined MLFQ after CTSS?",
    "Why was Corbato awarded the Turing Award?",
    "What is the significance of Multics in MLFQ history?",
    "Why does MLFQ assume new jobs might be short?",
    "What happens to long jobs over time in MLFQ?",
    "How does MLFQ avoid interactive job starvation?",
    "Why is starvation dangerous in a scheduling system?",
    "Why is security relevant in scheduling policies?",
    "How can malicious users exploit scheduling?",
    "What is the difference between Rule 4a and Rule 4b?",
    "Why were Rules 4a and 4b merged into Rule 4?",
    "How does I/O affect priority in MLFQ?",
    "Why should relinquishing CPU not always reset allotment?",
    "What happens in MLFQ if a process mixes CPU and I/O phases?",
    "Why must scheduling be secure against attacks?",
    "What happens when priority boost is too frequent?",
    "What happens when priority boost is too infrequent?",
    "Why is finding the right S value difficult?",
    "How does machine learning help tune scheduling?",
    "What does Figure 8.2 illustrate?",
    "What does Figure 8.3 illustrate?",
    "What does Figure 8.4 show with and without boosts?",
    "What does Figure 8.5 show about gaming?",
    "What does Figure 8.6 demonstrate about time slices?",
    "How does MLFQ differ from fixed-priority scheduling?",
    "Why is round robin used within queues?",
    "How does MLFQ ensure fairness?",
    "What happens to a job after being boosted?",
    "How does MLFQ handle mixed workloads?",
    "What is the role of history in MLFQ?",
    "Why does MLFQ provide good responsiveness?",
    "How does MLFQ prevent monopolization of CPU?",
    "Why does the OS rarely know job length?",
    "Why must MLFQ adapt dynamically?",
    "What is the advantage of multiple queues vs one?",
    "What is an allotment relative to time slice?",
    "How do queue lengths affect performance?",
    "Why is administrator tuning important in MLFQ?",
    "What is the relationship between MLFQ and SJF?",
    "How does MLFQ approximate STCF?",
    "What is the general principle behind feedback in scheduling?",
    "Why is advice optional for OS to follow?",
    "What are the five final rules of MLFQ?",
    "How does MLFQ prevent monopolization of CPU?",
    "How does MLFQ differ from priority aging?",
    "What trade-off does MLFQ make between fairness and efficiency?",
    "Why is feedback crucial for dynamic priority adjustment?",
    "How does MLFQ treat newly arriving jobs compared to old ones?",
    "Why can a newly arrived job preempt a running job in MLFQ?",
    "What happens if multiple new jobs arrive simultaneously?",
    "How can quantum length impact system responsiveness in MLFQ?",
    "What determines the number of queues in an MLFQ system?",
    "How do lower queues affect overall throughput?",
    "Why does MLFQ require timer interrupts for proper operation?",
    "How does MLFQ interact with context switching overhead?",
    "What is the benefit of exponential queue time slices in MLFQ?",
    "Why are higher queues granted shorter quanta?",
    "What information does MLFQ use to infer CPU-bound behavior?",
    "How does MLFQ prioritize I/O-bound processes?",
    "Why does MLFQ sometimes reset all priorities periodically?",
    "What performance metric does MLFQ primarily optimize?",
    "Why might MLFQ violate strict fairness?",
    "How does MLFQ adapt to changing workloads?",
    "What distinguishes interactive and batch workloads in MLFQ?",
    "Why can frequent priority boosting hurt throughput?",
    "How can an operating system tune MLFQ parameters automatically?",
    "What heuristics guide MLFQ queue promotion and demotion?",
    "Why is queue starvation a form of priority inversion?",
    "What happens if the boost interval is shorter than the quantum?",
    "How does MLFQ behave under heavy I/O workloads?",
    "Why is feedback scheduling called a 'learning algorithm'?",
    "What is the connection between MLFQ and adaptive scheduling?",
    "How does MLFQ compare to proportional share scheduling?",
    "What is the theoretical limit of responsiveness in MLFQ?",
    "How can MLFQ scheduling cause unfairness between CPU-bound tasks?",
    "Why must MLFQ track recent CPU usage per process?",
    "How is the 'recent CPU' metric updated in FreeBSD's scheduler?",
    "Why might MLFQ require kernel accounting support?",
    "How can starvation be mitigated without full priority boosts?",
    "Why does a fixed time quantum hurt short interactive tasks?",
    "What is the effect of too many queues in MLFQ?",
    "How can queue merging improve MLFQ efficiency?",
    "Why does Linux not use pure MLFQ anymore?",
    "What scheduling class in Linux approximates MLFQ behavior?",
    "Why is MLFQ considered a non-preemptive policy between boosts?",
    "What are the trade-offs between deterministic and heuristic boosts?",
    "Why does MLFQ sometimes need manual retuning?",
    "How can user 'nice' values interact with MLFQ levels?",
    "What does aging accomplish differently from boosting?",
    "Why can boosting reset the system’s learned behavior?",
    "How does MLFQ handle processes with unpredictable I/O?",
    "Why does queue depth affect context switch frequency?",
    "How can time slice granularity impact throughput in MLFQ?",
    "Why must MLFQ prevent CPU monopolization by compute jobs?",
    "What happens if all jobs are long and CPU-bound?",
    "Why does MLFQ perform better for mixed workloads?",
    "How can MLFQ scheduling affect cache locality?",
    "Why might MLFQ require hardware timer precision?",
    "How does load average influence queue movement?",
    "What makes tuning MLFQ empirically challenging?",
    "Why is simulation useful for designing MLFQ parameters?",
    "What role does user interactivity play in MLFQ design?",
    "Why can a static priority scheme underperform MLFQ?",
    "What is the complexity of MLFQ queue selection?",
    "How can priority inversion be detected in MLFQ?",
    "Why must MLFQ include fairness counters?",
    "What role does quantum expiration play in feedback accuracy?",
    "How can CPU bursts be predicted heuristically in MLFQ?",
    "Why does MLFQ sometimes revert to RR scheduling within queues?",
    "How can hybrid schedulers combine MLFQ and lottery scheduling?",
    "What are the limitations of using MLFQ in real-time systems?",
    "Why might an OS replace MLFQ with a CFS-like model?",
    "How does MLFQ affect power efficiency in CPUs?",
    "Why can longer quanta improve throughput but harm latency?",
    "How does MLFQ scheduling impact system fairness under load?",
    "What is the trade-off between adaptability and predictability in MLFQ?",
    "Why are decay-based priority systems considered more stable?",
    "What metrics determine MLFQ success for user workloads?",
    "Why do interactive processes benefit from frequent relinquishment?",
    "How does MLFQ recover from misclassification of jobs?",
    "What makes MLFQ non-deterministic compared to static scheduling?",
    "Why can time slicing reduce starvation but increase overhead?",
    "How can priority boosts mimic human 'patience decay' models?",
    "Why can over-tuning cause oscillations in MLFQ performance?",
    "How can feedback lag lead to poor queue assignment?",
    "Why is the concept of 'recent history' crucial in MLFQ?",
    "What is the impact of CPU burst variance on MLFQ accuracy?",
    "How can priority decay prevent starvation without full boosts?",
    "Why might I/O wait times be included in priority computation?",
    "What makes MLFQ ideal for desktop workloads?",
    "Why is MLFQ less suitable for predictable embedded systems?",
    "What aspects of MLFQ can be learned by reinforcement learning?",
    "How does MLFQ differ from deadline scheduling?",
    "Why can MLFQ mimic SJF in the long run?",
    "How can adaptive quanta improve scheduling stability?",
    "Why must context switch overhead be measured in MLFQ tuning?",
    "How can user behavior analytics improve MLFQ heuristics?",
    "Why does MLFQ rely heavily on empirical constants?",
    "What are the open research questions about MLFQ tuning?",
    "Why is MLFQ used as a teaching example in OS textbooks?",
    "How can modern cloud schedulers generalize MLFQ concepts?",
    "Why is MLFQ still relevant in multiprocessor environments?",
    "How can per-core MLFQ scheduling differ from global scheduling?",
    "What role does cache sharing play in MLFQ design?",
    "How can kernel developers benchmark MLFQ performance?",
    "Why might containerized workloads require modified MLFQ rules?",
    "What is the relationship between load balancing and MLFQ fairness?",
    "Why is per-thread priority tracking harder in MLFQ systems?",
    "How do hybrid schedulers combine MLFQ with priority inheritance?",
]

# # Create dataset
# dataset = [{"system": system_prompt, "user": q} for q in queries]

# # Save as JSON
# with open("custom_shared_prefix_dataset.json", "w") as f:
#     json.dump(dataset, f, indent=2)

# print(f"Dataset created with {len(dataset)} entries.")


# Create dataset entries (each question = one user prompt)
dataset = [{"id": i, "user": q} for i, q in enumerate(queries)]

# Write to JSON file
output_path = "custom_shared_prefix_dataset.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated dataset with {len(dataset)} prompts")
print(f"Saved to: {os.path.abspath(output_path)}")

