import requests
import threading
import time
import random

# --- Configuration ---
PORT = 30000
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NUM_PROMPTS_PER_FAMILY = 300  # you can increase this to saturate the GPU

CHAPTERS = {
    "Chapter 4": """
    In the study of operating systems, one of the most fundamental ideas is the concept of the process. This chapter from Operating Systems: Three Easy Pieces introduces the process abstraction as the core mechanism through which an operating system (OS) provides the illusion that multiple programs are running at the same time, even though the underlying machine might have only a single physical CPU. The chapter emphasizes that this illusion of concurrency, isolation, and independence is central to how modern computers operate. A process, in essence, is a running instance of a program — a dynamic entity that encapsulates both code and the current state of execution.

The authors begin by contrasting a program with a process. A program is a passive entity — a static collection of bytes stored on disk, containing instructions, data, and possibly some metadata. On its own, a program does nothing. It simply resides in the storage medium, waiting to be executed. A process, on the other hand, is the active form of that program. When a user runs a program, the operating system takes that stored representation and transforms it into a living entity in memory — one that can execute instructions, allocate and free memory, perform I/O, and interact with other processes. This transformation is not trivial; it requires the OS to allocate resources, initialize state, and manage execution context. Thus, a process can be viewed as a program in execution, along with all the associated information the OS maintains to keep it running correctly and efficiently.

At the heart of this chapter lies the question: How can an operating system make it appear that many programs are running simultaneously on a single CPU? The answer lies in virtualization of the CPU. The OS creates the illusion of multiple CPUs by rapidly switching between processes, giving each a short slice of CPU time. This technique is known as time-sharing. By running one process for a brief interval, saving its state, and then switching to another process, the OS makes each process believe that it has its own dedicated processor. This virtualization is achieved through a combination of hardware support (such as timer interrupts and context switching) and software policies (like scheduling algorithms). The process abstraction is the conceptual foundation that allows this virtualization to appear seamless to the user and to the running programs.

The definition of a process goes beyond simply saying “a running program.” A process must include everything necessary to fully describe its current execution. This includes several components. First, the address space, which represents all the memory a process can access. This address space is divided into regions: the code or text segment (containing the compiled instructions), the static data segment (containing global variables), the heap (where dynamically allocated memory resides), and the stack (used for local variables and function call frames). The OS ensures that each process has its own private address space so that one process cannot interfere with another’s memory. Second, a process includes the execution context — the values of the CPU’s registers, especially the program counter (which indicates the next instruction to execute) and the stack pointer (which points to the top of the call stack). Third, the process includes all other OS-maintained information such as open file descriptors, network connections, current working directory, environment variables, and accounting information. Together, this collection of data constitutes the process state.

The operating system keeps track of this information using an internal data structure called the Process Control Block (PCB). The PCB acts as a container for all the metadata the OS needs to manage a process. It stores the process’s current state (running, ready, or blocked), the contents of registers when the process is not executing, pointers to memory regions, scheduling information, and links to other processes (such as parent or child processes). Whenever a context switch occurs — that is, when the OS decides to stop running one process and start another — the OS saves the outgoing process’s state into its PCB and restores the state of the incoming process from its PCB. This mechanism allows multiple processes to appear as if they are executing concurrently even though only one can actually be executing at a time on a single CPU.

The chapter explains that the process abstraction relies on time-sharing and space-sharing. Time-sharing refers to the division of CPU time among multiple processes. Space-sharing refers to the division of physical resources like memory among processes. While the CPU can only execute one process at a time, different parts of physical memory can be allocated to different processes simultaneously. Thus, space-sharing allows coexistence in memory, while time-sharing enables the illusion of concurrent execution. Together, these techniques underpin multiprogramming — the ability of the OS to maintain multiple processes in memory and switch between them as needed.

The chapter then introduces the concept of process states. At any moment, a process can be in one of several states: running, ready, or blocked. A process is running when it currently occupies the CPU and is actively executing instructions. A process is ready when it could run if given the CPU but is currently waiting for its turn. A process is blocked when it is waiting for some event to occur, such as the completion of an I/O operation or the arrival of a signal. Processes often transition between these states. For example, a running process might issue a disk read operation and then become blocked, allowing the CPU to be given to another ready process. When the disk read completes, the blocked process moves back to the ready state and eventually to running again. This constant cycling between states is orchestrated by the OS scheduler, which determines which process should run next.

Understanding these state transitions is critical to understanding how an OS achieves concurrency. The chapter uses examples to show how a process alternates between computation and waiting for I/O. Since I/O operations are typically slow compared to CPU speed, it would be inefficient for the CPU to sit idle while waiting for one process’s I/O to finish. Instead, the OS allows another process to use the CPU during that time. This is the essence of overlapping computation and I/O, and it is one of the major benefits of process-based multitasking.

The chapter also details how a process is created. When a user launches a program, the operating system performs several steps. First, it loads the program’s executable file from disk into memory, mapping its code and static data into the process’s address space. It then allocates memory for the stack and heap, sets up necessary system resources, and initializes the CPU registers, including the program counter to point to the start of the program’s main function. The OS also sets up the standard input, output, and error streams so the process can interact with the user or other programs. Once everything is prepared, the OS places the process into the ready queue, and when the scheduler selects it, it begins execution.

When a process finishes execution or encounters a fatal error, it must be terminated. Termination involves freeing all the resources allocated to that process — including memory, file descriptors, and kernel data structures — and possibly notifying other processes (such as the parent) of its exit status. Some operating systems introduce an intermediate “zombie” state to represent processes that have finished execution but whose parent has not yet collected their termination status. This ensures that all resource cleanup is accounted for.

The authors emphasize that the process abstraction simplifies both system design and programming. Without the process abstraction, each program would need to be aware of and manage its own share of the CPU, coordinate with others to avoid conflicts, and handle all forms of I/O synchronization manually. This would make software development extraordinarily complex and error-prone. By letting the OS abstract away these concerns, processes can be written as if they are the only entities running. This simplicity allows for modularity, security, and stability. Each process is isolated from others; if one crashes, it does not typically affect others. The OS enforces protection boundaries between processes using hardware mechanisms such as separate address spaces, so a bug in one process cannot directly corrupt another’s memory.

A related topic discussed in this chapter is process isolation and protection. Each process operates within its own virtual address space. When the process tries to access memory, the memory management unit (MMU) translates the virtual address to a physical address. The OS configures the MMU so that a process can only access its own pages of memory. This hardware-enforced boundary ensures that processes cannot read or write to each other’s data. It is a crucial component of system security and stability. The OS also manages controlled sharing of data through mechanisms such as shared memory regions or interprocess communication (IPC) primitives, which allow processes to exchange information safely without violating isolation.

The authors further elaborate on the process life cycle. Every process begins in a “new” state, transitions to “ready” once initialized, moves to “running” when selected by the scheduler, may become “blocked” while waiting for I/O, and eventually terminates. These transitions are managed by the OS kernel, which maintains queues of processes in various states. The ready queue holds processes waiting for CPU time, while blocked queues hold those waiting on specific events. The scheduler’s job is to select from the ready queue and allocate CPU time fairly and efficiently. Later chapters of OSTEP explore the policies behind scheduling, but Chapter 4 focuses on establishing the conceptual framework that makes such scheduling meaningful.

The chapter introduces the mechanism of a context switch, which is central to process management. A context switch occurs when the OS switches from running one process to another. During this switch, the OS saves the current process’s CPU state — including the values of registers and the program counter — into its PCB and loads the state of the next process. Context switches can occur voluntarily, such as when a process blocks for I/O, or involuntarily, such as when a timer interrupt forces the OS to preempt a running process. Although context switches enable multitasking, they also introduce overhead because saving and restoring CPU state takes time. Hence, efficient context switching is an important performance consideration in OS design.

The chapter also distinguishes between mechanism and policy. Mechanism refers to the low-level operations that make process management possible — for example, the ability to perform a context switch or to add and remove processes from queues. Policy, on the other hand, refers to the decision-making logic that determines how these mechanisms are used — for instance, which process should run next or how long it should be allowed to run. Separating mechanism from policy allows for flexibility: the same OS kernel mechanisms can support different scheduling policies without rewriting the underlying code.

An interesting perspective that the chapter provides is historical. Early computers could run only one program at a time. Users had to load a program, wait for it to finish, and then load the next one. As computing evolved, the need for multiprogramming became apparent — allowing the CPU to switch between tasks so that one process could use the CPU while another waited for I/O. The process abstraction emerged from this need, providing a standardized way to represent running programs and manage their concurrent execution. Today, this abstraction remains fundamental, even though modern systems may have multiple CPUs or cores. Even on multicore systems, each core executes processes under the same principles, just with more true parallelism.

While the chapter’s focus is primarily on processes, it briefly hints at the concept of threads. Threads are essentially lightweight processes that share the same address space and resources but maintain separate registers and stack pointers. The process abstraction provides the foundation upon which threads are built. Understanding processes first makes it easier to grasp how multithreading works and how it extends the model to multiple execution streams within a single process.

The chapter ends by emphasizing the importance and elegance of the process abstraction. It encapsulates all the complexity of running a program — including CPU virtualization, memory management, and I/O coordination — behind a simple interface. To a user or application developer, a process appears as an isolated, independent unit of execution with its own CPU and memory. To the OS, a process is a data structure to be created, scheduled, suspended, and terminated. This dual perspective makes the abstraction both powerful and flexible. It allows the OS to maintain control while providing users with simplicity and reliability.

Finally, the authors underscore that while the abstraction is conceptually clean, the implementation is intricate. The OS must ensure fairness in CPU allocation, minimize context-switching overhead, handle synchronization between processes, and enforce protection boundaries. It must also maintain responsiveness to user input and handle long-running background jobs without starvation. All these concerns are orchestrated under the umbrella of the process abstraction, demonstrating the sophistication of operating system design.
    """,
    "Chapter 7": """
    Modern operating systems must manage multiple programs running seemingly simultaneously on a finite set of CPUs. The operating system achieves this through CPU scheduling, the process by which it decides which process or thread gets to execute on the CPU at any given time. Chapter 7 of Operating Systems: Three Easy Pieces explores CPU scheduling in depth, discussing both the conceptual principles and practical mechanisms that allow multitasking systems to operate efficiently, fairly, and responsively. CPU scheduling lies at the heart of process management because, while processes may compete for various resources such as memory, I/O devices, and network connections, the CPU is the most critical shared resource. Efficient scheduling maximizes system throughput, ensures responsiveness for interactive users, and maintains fairness among competing processes.

The chapter begins by explaining why scheduling is necessary. In a system with multiple processes, not all processes can run at once due to the limited number of CPUs. Even on a multicore system, the number of runnable processes often exceeds the number of available cores. Moreover, processes often alternate between CPU bursts, where they actively perform computation, and I/O bursts, where they wait for input or output. To maximize CPU utilization, the operating system must ensure that while one process waits for I/O, another process is using the CPU. Without proper scheduling, the CPU could remain idle, reducing throughput and efficiency. Thus, CPU scheduling is essential for leveraging hardware resources effectively and providing the illusion of concurrent execution for multiple processes.

The authors introduce the key objectives of CPU scheduling. These objectives include maximizing CPU utilization, improving throughput, minimizing turnaround time, and providing responsiveness for interactive users. CPU utilization refers to the percentage of time the CPU is actively executing processes rather than idling. Throughput measures the number of processes completed per unit of time. Turnaround time is the total time from process submission to completion, and response time is the time between a request and the first response produced by the system. Achieving a balance among these metrics is nontrivial because optimizing one metric may adversely affect another. For example, maximizing throughput by favoring long-running batch processes could increase the response time for interactive tasks. CPU scheduling algorithms aim to strike a balance based on the system’s goals and the types of workloads it serves.

To understand scheduling decisions, the chapter introduces the concept of CPU bursts and I/O bursts. Most processes alternate between periods of CPU-bound computation and I/O-bound waiting. CPU-bound processes perform long computations with infrequent I/O, while I/O-bound processes frequently request I/O operations, often completing short CPU bursts. The operating system can exploit this pattern by scheduling CPU-bound processes and I/O-bound processes differently. For instance, prioritizing I/O-bound processes can keep the CPU busy while long-running CPU-bound processes wait, thereby improving overall system responsiveness.

The chapter classifies CPU scheduling into non-preemptive and preemptive policies. In non-preemptive scheduling, once a process begins execution on the CPU, it runs to completion of its current CPU burst or until it voluntarily yields the CPU, such as by performing I/O. Examples of non-preemptive algorithms include First-Come, First-Served (FCFS) and Shortest Job First (SJF). In preemptive scheduling, the OS can interrupt a running process to assign the CPU to another process. Preemption is often triggered by timer interrupts, allowing the OS to enforce fairness, prioritize interactive tasks, and respond to changing workloads. Preemptive algorithms include Round-Robin (RR), Preemptive SJF (also called Shortest Remaining Time First, SRTF), and Priority Scheduling.

The chapter explains First-Come, First-Served (FCFS) scheduling in detail. FCFS is conceptually simple: the CPU is assigned to processes in the order they arrive in the ready queue. While easy to implement, FCFS has several limitations. One major drawback is the convoy effect, where short processes wait behind long-running processes, increasing average turnaround time and reducing system responsiveness. For example, if a long CPU-bound process arrives before several short I/O-bound processes, all the short processes must wait, despite being able to complete quickly. FCFS may be suitable for batch systems but is often inadequate for interactive systems where responsiveness is crucial.

Shortest Job First (SJF) scheduling attempts to minimize average turnaround time by selecting the process with the shortest next CPU burst. In theory, SJF is optimal for minimizing average turnaround time among all non-preemptive algorithms. However, SJF requires knowledge of future CPU bursts, which is not usually available. The chapter discusses practical approximations using exponential averaging, where the OS predicts the next CPU burst based on the history of previous bursts. Preemptive SJF, also called Shortest Remaining Time First (SRTF), allows the OS to preempt a running process if a newly arriving process has a shorter predicted CPU burst. This approach further reduces average turnaround time but introduces the overhead of frequent context switches and the potential for starvation of longer processes.

The authors then explore priority-based scheduling, where each process is assigned a priority, and the CPU is allocated to the highest-priority process. Priority scheduling can be either preemptive or non-preemptive. Preemptive priority scheduling interrupts a running process if a higher-priority process becomes ready. While priority scheduling allows the system to favor important tasks, it introduces the risk of starvation for low-priority processes. To address starvation, operating systems may implement aging, gradually increasing the priority of processes that wait in the ready queue for an extended period. This ensures that all processes eventually receive CPU time while still favoring higher-priority tasks.

Round-Robin (RR) scheduling is a widely used preemptive algorithm, especially in interactive systems. In RR, each process is assigned a fixed time slice or quantum. The CPU is given to the first process in the ready queue; if the process does not complete within its time quantum, it is preempted and placed at the end of the queue. RR ensures fairness and responsiveness because all processes get regular access to the CPU. The choice of quantum size is critical: a very short quantum increases context-switching overhead, while a very long quantum reduces responsiveness. RR is particularly effective when combined with multiple queues or dynamic priority adjustments to accommodate different workload types.

The chapter also introduces multilevel queue scheduling, which partitions the ready queue into multiple queues based on process characteristics. For instance, interactive processes may reside in a high-priority queue, while CPU-bound batch processes occupy a lower-priority queue. Each queue may use its own scheduling algorithm, and processes rarely migrate between queues. This approach allows the system to optimize performance for different types of workloads. A variation, multilevel feedback queue scheduling, allows processes to move between queues based on their observed behavior, such as CPU burst length or waiting time. This dynamic adjustment improves fairness and prevents starvation while maintaining responsiveness.

To evaluate scheduling algorithms, the chapter presents performance metrics, including turnaround time, waiting time, response time, and CPU utilization. Turnaround time measures the total time from submission to completion, providing a sense of how long a process must wait. Waiting time measures how long a process spends in the ready queue, which is particularly relevant for interactive systems. Response time, the interval from submission to the first response, is crucial for user-perceived performance. CPU utilization measures how efficiently the CPU is used. Different scheduling algorithms affect these metrics in different ways, and trade-offs must be carefully considered. For example, SJF minimizes average turnaround time but may increase response time for some processes, while RR improves responsiveness but can increase overhead.

The authors illustrate scheduling concepts using Gantt charts and worked examples. Gantt charts visually represent which process is running on the CPU at each time interval. By examining these charts, one can calculate turnaround time, waiting time, and response time for each process. These examples highlight the practical implications of different scheduling choices, showing how the order of execution affects performance metrics. They also demonstrate the impact of preemption, time quantum size, and priority assignment on system behavior.

Starvation and fairness are recurring themes in CPU scheduling. Algorithms that favor short or high-priority processes can cause longer or lower-priority processes to wait indefinitely. This is particularly problematic in systems with strict fairness or real-time requirements. Aging, feedback queues, and careful priority assignment are common solutions to mitigate starvation. The chapter emphasizes that a scheduler must balance fairness with efficiency: the system should not only maximize CPU utilization but also ensure that all processes receive a reasonable share of CPU time.

The chapter concludes with a discussion of real-world considerations. While theoretical algorithms like SJF may minimize turnaround time in idealized scenarios, practical operating systems must account for dynamic workloads, unpredictable process behavior, hardware constraints, and system responsiveness. Modern OS schedulers are often hybrid, combining multiple algorithms to handle different workloads. For example, interactive tasks may use RR with a small time quantum, batch jobs may use FCFS or multilevel queues, and I/O-bound processes may receive priority boosts to improve throughput. Additionally, context switch overhead, cache effects, and the presence of multiple cores complicate scheduling decisions. The chapter emphasizes that effective CPU scheduling requires both solid conceptual understanding and careful tuning to meet system goals.

In summary, Chapter 7 of OSTEP provides a comprehensive understanding of CPU scheduling. It defines the purpose of scheduling, explains the trade-offs involved, and categorizes algorithms into non-preemptive and preemptive types. The chapter explores FCFS, SJF, priority scheduling, round-robin, and multilevel queue approaches, illustrating each with examples and performance metrics. It addresses starvation, fairness, and real-world considerations, highlighting the challenges and solutions that operating systems employ to manage CPU resources efficiently. Understanding CPU scheduling is essential not only for OS design but also for evaluating and improving the performance of applications and systems.

The chapter emphasizes that CPU scheduling is more than a theoretical exercise; it directly affects user experience, system throughput, and resource utilization. By carefully selecting algorithms and tuning parameters such as time quantum, the operating system can provide responsive, fair, and efficient execution for a diverse set of processes. Whether dealing with interactive desktops, high-performance servers, or real-time systems, the principles of CPU scheduling guide how the CPU is allocated and managed, shaping the performance and behavior of the entire system. Moreover, the concepts introduced in this chapter serve as the foundation for more advanced topics such as multithreading, multiprocessor scheduling, and real-time operating systems, where the complexity and importance of scheduling decisions increase significantly.

CPU scheduling also serves as a practical illustration of the broader theme of mechanism versus policy. The mechanism includes the OS’s ability to perform context switches, maintain queues, and track process states. The policy dictates how those mechanisms are used to allocate CPU time — which process to run next, for how long, and under what conditions to preempt. By separating mechanism and policy, operating systems achieve flexibility, allowing schedulers to be tailored to
    """,
    "Chapter 39": """
   Files and directories are foundational abstractions in modern operating systems, providing a structured and convenient way to store, organize, and retrieve data. Chapter 39 of Operating Systems: Three Easy Pieces explores these abstractions, explaining how the operating system presents persistent storage to users and applications in a coherent and manageable form. Unlike volatile memory, where data disappears when power is lost, files and directories reside on persistent storage devices such as hard drives, solid-state drives, or network-attached storage. The OS abstracts away the details of physical storage and presents a logical, uniform interface for creating, accessing, modifying, and organizing data. This abstraction is essential because raw hardware provides little in the way of structure or usability.

At the heart of this chapter is the file abstraction. A file is essentially a named sequence of bytes that can be read or written by a program. This abstraction hides the complexities of underlying storage, such as block allocation, fragmentation, and device characteristics. From the perspective of a user or application, a file is a simple entity: it has a name, can contain data, and can be accessed in various ways. The operating system manages the details of where the file’s contents reside on disk, how to handle simultaneous access by multiple processes, and how to ensure data integrity even in the presence of failures. The simplicity of the file abstraction is what allows programmers to interact with storage without worrying about the underlying hardware mechanics.

The chapter explains that files are stored on disks in blocks, which are fixed-size chunks of storage. The OS maps a file’s logical sequence of bytes into one or more disk blocks. This mapping may not be contiguous, as files can grow or shrink over time, leading to fragmented storage. To manage this, file systems maintain metadata structures such as inodes, which track the locations of a file’s blocks, ownership information, permissions, timestamps, and other attributes. Inodes serve as the central data structure for file management in UNIX-like systems, allowing the OS to efficiently locate a file’s contents and enforce access controls. Other file system designs may use different metadata structures, but the principle remains the same: the OS maintains information about both the file’s contents and its properties.

Files can be accessed in different modes, including sequential and random access. Sequential access reads or writes data in a linear order from the beginning to the end of the file, while random access allows the program to read or write at arbitrary positions within the file. To support these modes, the OS maintains a file descriptor or file handle for each open file, which includes the current position within the file and the mode of access. When a process reads or writes, the OS uses the descriptor to locate the correct blocks and update the file pointer. File descriptors abstract away the underlying complexity of block allocation and physical storage, providing a uniform interface for programmatic file operations.

The chapter highlights the importance of directories in organizing files. A directory is itself a file that contains mappings between file names and metadata such as inodes. Directories provide a hierarchical structure, allowing users and programs to organize files in a tree-like manner. This hierarchy supports efficient navigation, avoids naming conflicts, and provides context for access permissions. For example, a user may have a “Documents” directory containing files for work and a “Pictures” directory containing media files. Directories can contain other directories, forming a nested structure known as a directory tree. This organization simplifies both human understanding and programmatic file management.

Directories also play a critical role in file path resolution. A path is a sequence of directory names leading to a file, and it can be absolute (starting from the root directory) or relative (starting from the current working directory). The operating system traverses the directory hierarchy to locate a file based on its path. During this traversal, the OS checks permissions at each level, ensuring that a process can only access directories and files for which it has the appropriate rights. This hierarchical namespace and access control mechanism underpin security and organization in modern file systems.

The chapter emphasizes file system operations such as creation, deletion, reading, writing, and renaming of files. File creation involves allocating a new inode, initializing metadata, and updating the parent directory to include the new file name. File deletion requires removing the directory entry and deallocating the file’s blocks, while ensuring that any open file descriptors still function correctly until the file is fully closed. Reading and writing involve translating logical offsets into disk block locations, handling partial blocks, and possibly caching data in memory to improve performance. The chapter also discusses how operations are designed to be atomic and consistent, minimizing the risk of corruption in case of crashes or power failures.

Another key concept is file permissions and access control. Modern operating systems enforce security by associating ownership and access rights with each file. Typical permissions include read, write, and execute for the file owner, a group of users, and others. These permissions are enforced by the OS whenever a process attempts to access a file, preventing unauthorized reading, modification, or execution. File systems may also support more advanced access control mechanisms, such as Access Control Lists (ACLs), which provide fine-grained permission specifications beyond the simple owner/group/other model.

The chapter also covers special files and their roles. Not all files correspond to regular data; some represent devices, pipes, or sockets. Device files allow processes to interact with hardware devices through the file abstraction. For example, reading from a device file might return input from a keyboard, while writing to a device file could send data to a printer. Pipes and sockets provide mechanisms for interprocess communication using the familiar file interface. By unifying diverse operations under the file abstraction, the operating system simplifies programming and enhances modularity.

File system performance is another focus of the chapter. Efficient file access depends on minimizing disk seeks and maximizing throughput. To achieve this, operating systems employ techniques such as buffering, caching, and read-ahead. Buffering temporarily holds data in memory to allow processes to access it without waiting for disk I/O. Caching keeps frequently accessed data in memory to reduce disk access latency. Read-ahead anticipates sequential file access and prefetches blocks into memory. Together, these techniques significantly improve performance while maintaining the abstraction of a simple, uniform file interface.

The chapter explores consistency and reliability concerns in file systems. Since files represent persistent data, it is critical to ensure that updates are applied correctly, even in the event of crashes. Modern file systems implement journaling or log-based updates to track pending changes. A journal records intended modifications before applying them, allowing the system to recover consistently in case of a failure. File systems may also perform atomic operations, ensuring that complex updates either complete fully or not at all, preventing partial writes and corruption.

Naming and symbolic links are also discussed. Files can have multiple names, and symbolic links allow one file to point to another. This capability enables flexible organization, aliasing, and sharing of resources without duplicating data. Hard links map multiple names to the same inode, while symbolic links store a path reference to another file. Both mechanisms enrich the namespace but introduce additional complexity in path resolution and access control.

The chapter further considers directory operations, such as creation, deletion, renaming, and listing contents. Like files, directories maintain metadata and must be updated consistently. The OS must ensure that operations like removing a directory are safe, preventing deletion if it contains files or subdirectories, unless explicitly requested. Directory traversal, which occurs when resolving paths, is a fundamental operation that impacts system performance and security.

Finally, the chapter discusses the interplay between files, directories, and the process abstraction. Processes interact with files through file descriptors and system calls such as open, read, write, close, and unlink. The operating system maintains internal tables linking file descriptors to inodes and tracks open files per process. When multiple processes access the same file concurrently, the OS enforces consistency and synchronization, ensuring that data is not corrupted. This interplay illustrates the broader principle of abstraction in operating systems: complex operations on hardware are hidden behind simple, uniform interfaces for processes to use reliably and safely.

In conclusion, Chapter 39 of OSTEP provides a comprehensive view of files and directories as essential abstractions in modern operating systems. Files provide a uniform interface for persistent storage, hiding the complexities of physical media, block allocation, and device management. Directories organize files hierarchically, supporting efficient navigation, access control, and namespace management. The chapter explores operations on files and directories, access control, special files, performance optimizations, consistency mechanisms, and the integration of these abstractions with the process interface. Understanding these concepts is critical for both system designers and application developers, as they form the foundation for reliable, secure, and efficient data storage and management.

By providing persistent storage, structure, and controlled access, files and directories enable applications to maintain state, share information, and interact with the system effectively. The abstractions described in this chapter illustrate the elegance of operating system design: complex, failure-prone, and hardware-dependent operations are transformed into simple, predictable, and uniform interfaces. These abstractions, combined with processes, memory, and CPU scheduling, create a coherent and powerful environment in which modern computing systems operate, allowing users and applications to focus on higher-level functionality without worrying about low-level details.
    """
}

# ------------------ PROMPT GENERATION ------------------

PROMPT_TEMPLATES = [
    "Explain {} in simple terms.",
    "Why is {} important in operating systems?",
    "Give a real-world example related to {}.",
    "What are some challenges in {}?",
    "How does the OS handle {} internally?",
    "Write a short paragraph describing {}.",
    "Summarize {} in two lines.",
    "List the key mechanisms involved in {}.",
    "What happens during {} at runtime?",
    "Compare {} with other OS mechanisms.",
]

# Define chapter-specific topics
CHAPTER_TOPICS = {
    "Chapter 4": [
        "process creation", "process termination", "context switching",
        "process states (running, ready, blocked)", "CPU virtualization",
        "process control block (PCB)", "mechanism vs policy in process scheduling",
        "process isolation and protection"
    ],
    "Chapter 7": [
        "First-Come, First-Served scheduling", "Shortest Job First scheduling",
        "Round-Robin scheduling", "priority scheduling", "preemption vs non-preemption",
        "CPU bursts vs I/O bursts", "turnaround time, waiting time, response time",
        "multilevel queue and multilevel feedback queue scheduling"
    ],
    "Chapter 39": [
        "file abstraction", "directory hierarchy", "inode structure",
        "file access modes (sequential and random)", "file descriptors",
        "file permissions and access control", "special files (devices, pipes, sockets)",
        "file system consistency and journaling"
    ]
}

def generate_prompts_from_chapters(num_per_chapter=100):
    prompts = []
    for chapter in CHAPTERS.keys():
        topics = CHAPTER_TOPICS[chapter]
        for _ in range(num_per_chapter):
            topic = random.choice(topics)
            prompt_template = random.choice(PROMPT_TEMPLATES)
            prompt = prompt_template.format(topic)
            prompts.append((chapter, prompt))
    return prompts


# ------------------ API REQUEST FUNCTIONS ------------------

def send_request(chapter, prompt, request_id):
    """Sends a single request to the LLM server and prints the response."""
    print(f"[{chapter} | Req {request_id}] Sending prompt: '{prompt[:40]}...'")

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"You are an OS expert discussing {chapter}."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 700  # Limit to ~100 words
    }

    try:
        response = requests.post(URL, json=data)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"\n--- [Reply for {chapter} | Req {request_id}] ---\n{reply}\n---------------------------\n")
        else:
            print(f"\n--- [Error for {chapter} | Req {request_id}] ---")
            print(f"Status code: {response.status_code}")
            print(f"Error details: {response.text}")
            print("---------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n--- [Connection Error for {chapter} | Req {request_id}] ---")
        print(f"Could not connect to the server at {URL}.")
        print(f"Please ensure your sglang server is running on port {PORT}.")
        print(f"Error: {e}")
        print("---------------------------\n")

# ------------------ BENCHMARK DRIVER ------------------

# --- Benchmark execution ---
if __name__ == "__main__":
    all_prompts = generate_prompts_from_chapters(NUM_PROMPTS_PER_FAMILY)
    total_reqs = len(all_prompts)

    print(f"Starting benchmark with {total_reqs} concurrent requests across {len(CHAPTERS)} chapters...")
    start_time = time.time()

    threads = []
    for i, (chapter, prompt) in enumerate(all_prompts):
        thread = threading.Thread(target=send_request, args=(chapter, prompt, i + 1))
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    end_time = time.time()
    elapsed = end_time - start_time
    rps = total_reqs / elapsed

    print("\n==============================")
    print(f"Completed {total_reqs} requests in {elapsed:.2f}s")
    print(f"Throughput: {rps:.2f} requests/sec")
    print("==============================")