/* platform.h - thin OS abstraction: file mapping, threads, atomics, timing, CPU info.
 * No third-party dependencies: Win32 API on Windows, POSIX elsewhere. */
#ifndef PITHOS_PLATFORM_H
#define PITHOS_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PITHOS_INLINE static __forceinline
#define PITHOS_ALIGN(n) __declspec(align(n))
#define PITHOS_THREAD_LOCAL __declspec(thread)
#else
#define PITHOS_INLINE static inline __attribute__((always_inline))
#define PITHOS_ALIGN(n) __attribute__((aligned(n)))
#define PITHOS_THREAD_LOCAL _Thread_local
#endif

/* ---- atomics (minimal set we need) ---- */
#if defined(_MSC_VER) && !defined(__clang__)
PITHOS_INLINE int  atomic_fetch_add_i32(volatile long *p, int v) { return _InterlockedExchangeAdd(p, v); }
PITHOS_INLINE int  atomic_load_i32(volatile long *p) { long v = *p; _ReadWriteBarrier(); return v; }
PITHOS_INLINE void atomic_store_i32(volatile long *p, int v) { _ReadWriteBarrier(); _InterlockedExchange(p, v); }
PITHOS_INLINE void cpu_relax(void) { _mm_pause(); }
typedef volatile long atomic_i32;
#else
PITHOS_INLINE int  atomic_fetch_add_i32(volatile int *p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }
PITHOS_INLINE int  atomic_load_i32(volatile int *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
PITHOS_INLINE void atomic_store_i32(volatile int *p, int v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#if defined(__x86_64__) || defined(__i386__)
PITHOS_INLINE void cpu_relax(void) { __builtin_ia32_pause(); }
#elif defined(__aarch64__)
PITHOS_INLINE void cpu_relax(void) { __asm__ __volatile__("yield"); }
#else
PITHOS_INLINE void cpu_relax(void) { }
#endif
typedef volatile int atomic_i32;
#endif

/* ---- read-only file mapping ---- */
typedef struct {
    const uint8_t *data;
    uint64_t size;
    void *h_file, *h_map;   /* Windows handles */
    int fd;                 /* POSIX */
} MappedFile;

int  map_file(MappedFile *mf, const char *path);
void unmap_file(MappedFile *mf);
/* Hint the OS that [addr, addr+len) will be needed soon (async read-ahead). */
void prefetch_memory(const void *addr, size_t len);
/* Tell the OS that [addr, addr+len) of a read-only file mapping is not needed for now:
 * the pages leave the working set (they stay in the file cache while RAM allows). */
void release_memory(const void *addr, size_t len);
/* Try to lock pages in RAM (best effort). Returns 1 on success. */
int  lock_memory(const void *addr, size_t len);

/* Background read-ahead: a helper thread that issues prefetch_memory() for queued ranges,
 * so compute threads never block on the request. Pushing never blocks (drops when full). */
typedef struct Prefetcher Prefetcher;
Prefetcher *prefetcher_create(void);
void prefetcher_push(Prefetcher *p, const void *addr, size_t len);
/* Drop all pending (not yet started) requests: stale read-ahead only competes with real work. */
void prefetcher_clear(Prefetcher *p);
void prefetcher_destroy(Prefetcher *p);

/* Generic helper thread + lock/condition pair (used by the GPU uploader). */
typedef struct PithosThread PithosThread;
PithosThread *thread_create(void (*fn)(void *), void *arg);
void thread_join(PithosThread *t);
typedef struct PithosLock PithosLock;
PithosLock *lock_create(void);
void lock_destroy(PithosLock *l);
void lock_acquire(PithosLock *l);
void lock_release(PithosLock *l);
void lock_wait(PithosLock *l);        /* must hold the lock */
void lock_notify_all(PithosLock *l);
int  lock_wait_ms(PithosLock *l, int ms);   /* must hold the lock; 0 on timeout */

/* Start a detached thread running fn(arg). Returns 0 on failure. */
int thread_spawn(void (*fn)(void *), void *arg);

/* ---- files and directories (UTF-8 paths on every OS) ---- */
#include <stdio.h>
FILE *file_open(const char *path, const char *mode);
int   file_remove(const char *path);
int   file_rename(const char *from, const char *to);   /* replaces `to` */
int   dir_create(const char *path);                    /* 1 if it exists afterwards */
/* Calls cb for every regular file name in dir (no path). Returns 0 if dir cannot be read. */
int   dir_list(const char *dir, void (*cb)(const char *name, void *u), void *u);
/* Like dir_list but also directories (hidden entries skipped), with a flag and the file size. */
int   dir_entries(const char *dir, void (*cb)(const char *name, int is_dir, uint64_t size, void *u), void *u);

/* ---- console ---- */
/* Read one line of UTF-8 input (newline removed). Returns 0 on end of input or when Ctrl+C
 * interrupts the read. */
int  console_read_line(char *buf, int cap);
/* Width of the terminal in columns (80 when unknown). */
int  console_columns(void);
/* Ctrl+C calls fn (from a signal handler or a system thread: it should only set flags)
 * instead of ending the process. */
void console_on_interrupt(void (*fn)(void));

/* Current folder as an absolute UTF-8 path. Returns 0 on failure. */
int  dir_current(char *out, int cap);

/* ---- dynamic libraries ---- */
void *lib_open(const char *name);
void *lib_sym(void *lib, const char *name);

/* ---- aligned allocation ---- */
void *aligned_malloc(size_t size, size_t align);
void  aligned_free(void *p);

/* Process page-fault counter and resident set size (0 if unavailable). */
void proc_mem_stats(uint64_t *page_faults, uint64_t *resident_bytes);

/* ---- time ---- */
double time_sec(void);

/* ---- system info ---- */
typedef struct {
    int has_avx2, has_fma, has_f16c, has_avx512f, has_avx512bw, has_avx512vnni, has_avxvnni;
    int has_neon, has_dotprod;
    int n_logical, n_physical;
    uint64_t total_ram, avail_ram;
    char brand[64];
} SysInfo;
void sys_info(SysInfo *si);
/* Operating system for humans and the agent, e.g. "Windows 11 Home 24H2 (version 10.0, build 26200), x64"
 * or "Ubuntu 24.04 LTS (Linux 6.8.0), x86_64". */
void os_describe(char *out, int cap);

/* ---- threads ---- */
typedef struct ThreadPool ThreadPool;
typedef void (*task_fn)(void *ctx, int task, int thread_id);

ThreadPool *pool_create(int n_threads);
void        pool_destroy(ThreadPool *p);
int         pool_size(const ThreadPool *p);
/* Run fn(ctx, i, tid) for i in [0, n_tasks), dynamically scheduled over all threads
 * (the caller participates as thread 0). Returns when all tasks are done. */
void        pool_run(ThreadPool *p, task_fn fn, void *ctx, int n_tasks);

#endif
