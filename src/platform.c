/* platform.c - see platform.h */
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define PITHOS_X86 1
#if !defined(_MSC_VER) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

/* ------------------------------------------------------------------ file mapping */

#if defined(_WIN32)
int map_file(MappedFile *mf, const char *path) {
    memset(mf, 0, sizeof(*mf));
    wchar_t wpath[1024];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024)) return 0;
    HANDLE hf = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (hf == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return 0; }
    HANDLE hm = CreateFileMappingW(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); return 0; }
    void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hm); CloseHandle(hf); return 0; }
    mf->data = (const uint8_t *)p;
    mf->size = (uint64_t)sz.QuadPart;
    mf->h_file = hf;
    mf->h_map = hm;
    return 1;
}

void unmap_file(MappedFile *mf) {
    if (mf->data) UnmapViewOfFile(mf->data);
    if (mf->h_map) CloseHandle((HANDLE)mf->h_map);
    if (mf->h_file) CloseHandle((HANDLE)mf->h_file);
    memset(mf, 0, sizeof(*mf));
}

/* declared locally: older SDK headers (or _WIN32_WINNT < 0x0602) lack it; resolved at runtime */
typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } pithos_mem_range;
typedef BOOL (WINAPI *PrefetchVirtualMemory_t)(HANDLE, ULONG_PTR, pithos_mem_range *, ULONG);

void prefetch_memory(const void *addr, size_t len) {
    static PrefetchVirtualMemory_t fn = NULL;
    static int resolved = 0;
    if (!resolved) {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        if (k32) fn = (PrefetchVirtualMemory_t)(void *)GetProcAddress(k32, "PrefetchVirtualMemory");
        resolved = 1;
    }
    if (!fn || !len) return;
    pithos_mem_range e;
    e.VirtualAddress = (PVOID)addr;
    e.NumberOfBytes = len;
    fn(GetCurrentProcess(), 1, &e, 0);
}

void release_memory(const void *addr, size_t len) {
    /* VirtualUnlock on pages that are not locked removes them from the working set
     * (documented behaviour); file-backed pages move to the standby list. */
    const uintptr_t pg = 4096;
    uintptr_t a = ((uintptr_t)addr + pg - 1) & ~(pg - 1);
    uintptr_t e = ((uintptr_t)addr + len) & ~(pg - 1);
    if (e > a) VirtualUnlock((LPVOID)a, (SIZE_T)(e - a));
}

int lock_memory(const void *addr, size_t len) {
    SIZE_T mn, mx;
    HANDLE proc = GetCurrentProcess();
    if (GetProcessWorkingSetSize(proc, &mn, &mx)) {
        SetProcessWorkingSetSize(proc, mn + len, mx + len);
    }
    return VirtualLock((LPVOID)addr, len) ? 1 : 0;
}
#else
int map_file(MappedFile *mf, const char *path) {
    memset(mf, 0, sizeof(*mf));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return 0; }
#ifdef MADV_RANDOM
    madvise(p, (size_t)st.st_size, MADV_RANDOM);
#endif
    mf->data = (const uint8_t *)p;
    mf->size = (uint64_t)st.st_size;
    mf->fd = fd;
    return 1;
}

void unmap_file(MappedFile *mf) {
    if (mf->data) munmap((void *)mf->data, (size_t)mf->size);
    if (mf->fd > 0) close(mf->fd);
    memset(mf, 0, sizeof(*mf));
}

void prefetch_memory(const void *addr, size_t len) {
#ifdef MADV_WILLNEED
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)addr & ~(uintptr_t)(pg - 1);
    madvise((void *)a, len + ((uintptr_t)addr - a), MADV_WILLNEED);
#else
    (void)addr; (void)len;
#endif
}

void release_memory(const void *addr, size_t len) {
#ifdef MADV_DONTNEED
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t a = ((uintptr_t)addr + pg - 1) & ~(uintptr_t)(pg - 1);
    uintptr_t e = ((uintptr_t)addr + len) & ~(uintptr_t)(pg - 1);
    if (e > a) madvise((void *)a, e - a, MADV_DONTNEED);
#else
    (void)addr; (void)len;
#endif
}

int lock_memory(const void *addr, size_t len) {
    return mlock(addr, len) == 0;
}
#endif

/* ------------------------------------------------------------------ memory */

void *aligned_malloc(size_t size, size_t align) {
#if defined(_WIN32)
    return _aligned_malloc(size ? size : 1, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
    return p;
#endif
}

void aligned_free(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

#if defined(_WIN32)
typedef struct {
    DWORD cb, PageFaultCount;
    SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
        QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage;
} pithos_pmc;
typedef BOOL (WINAPI *GetProcessMemoryInfo_t)(HANDLE, pithos_pmc *, DWORD);
#endif

void proc_mem_stats(uint64_t *page_faults, uint64_t *resident_bytes) {
    *page_faults = 0;
    *resident_bytes = 0;
#if defined(_WIN32)
    static GetProcessMemoryInfo_t fn = NULL;
    static int resolved = 0;
    if (!resolved) {
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        if (k32) fn = (GetProcessMemoryInfo_t)(void *)GetProcAddress(k32, "K32GetProcessMemoryInfo");
        resolved = 1;
    }
    pithos_pmc pmc;
    memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (fn && fn(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        *page_faults = pmc.PageFaultCount;
        *resident_bytes = pmc.WorkingSetSize;
    }
#elif defined(__linux__)
    FILE *f = fopen("/proc/self/stat", "r");
    if (f) {
        unsigned long minflt = 0, majflt = 0;
        long rss = 0;
        if (fscanf(f, "%*d %*s %*c %*d %*d %*d %*d %*d %*u %lu %*u %lu %*u %*u %*u %*d %*d %*d %*d %*d %*d %*u %*u %ld",
                   &minflt, &majflt, &rss) == 3) {
            *page_faults = minflt + majflt;
            *resident_bytes = (uint64_t)rss * (uint64_t)sysconf(_SC_PAGESIZE);
        }
        fclose(f);
    }
#endif
}

/* ------------------------------------------------------------------ time */

double time_sec(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ------------------------------------------------------------------ console */

static void (*g_on_interrupt)(void);

#if defined(_WIN32)
int console_read_line(char *buf, int cap) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        static wchar_t wbuf[16384];
        DWORD nread = 0;
        if (!ReadConsoleW(h, wbuf, 16383, &nread, NULL) || nread == 0) return 0;
        while (nread > 0 && (wbuf[nread - 1] == L'\n' || wbuf[nread - 1] == L'\r')) nread--;
        if (nread == 1 && wbuf[0] == 26) return 0; /* Ctrl+Z */
        int n = WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)nread, buf, cap - 1, NULL, NULL);
        buf[n] = 0;
        return 1;
    }
    if (!fgets(buf, cap, stdin)) return 0;
    size_t l = strlen(buf);
    while (l > 0 && (buf[l - 1] == '\n' || buf[l - 1] == '\r')) buf[--l] = 0;
    return 1;
}

int dir_current(char *out, int cap) {
    wchar_t w[2048];
    DWORD n = GetCurrentDirectoryW(2048, w);
    return n > 0 && n < 2048 && WideCharToMultiByte(CP_UTF8, 0, w, -1, out, cap, NULL, NULL) > 0;
}

int console_columns(void) {
    CONSOLE_SCREEN_BUFFER_INFO ci;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &ci)) return ci.srWindow.Right - ci.srWindow.Left + 1;
    return 80;
}

static BOOL WINAPI ctrl_handler(DWORD type) {
    if ((type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) && g_on_interrupt) {
        g_on_interrupt();
        return TRUE;
    }
    return FALSE;
}

void console_on_interrupt(void (*fn)(void)) {
    g_on_interrupt = fn;
    SetConsoleCtrlHandler(ctrl_handler, fn != NULL);
}
#else
#include <signal.h>
#include <sys/ioctl.h>
int dir_current(char *out, int cap) { return getcwd(out, (size_t)cap) != NULL; }

int console_columns(void) {
    struct winsize ws;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    const char *c = getenv("COLUMNS");
    return c && atoi(c) > 0 ? atoi(c) : 80;
}

int console_read_line(char *buf, int cap) {
    if (!fgets(buf, cap, stdin)) { clearerr(stdin); return 0; }
    size_t l = strlen(buf);
    while (l > 0 && (buf[l - 1] == '\n' || buf[l - 1] == '\r')) buf[--l] = 0;
    return 1;
}

static void sigint_handler(int sig) {
    (void)sig;
    if (g_on_interrupt) g_on_interrupt();
}

void console_on_interrupt(void (*fn)(void)) {
    g_on_interrupt = fn;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = fn ? sigint_handler : SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; /* no SA_RESTART: a pending read returns, so the prompt can react */
    sigaction(SIGINT, &sa, NULL);
}
#endif

/* ------------------------------------------------------------------ operating system */

#if defined(_WIN32)
#ifdef _MSC_VER
#pragma comment(lib, "advapi32.lib")
#endif
static int reg_str(const wchar_t *name, char *out, int cap) {
    wchar_t w[256];
    DWORD sz = sizeof(w);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", name, RRF_RT_REG_SZ, NULL,
                     w, &sz) != ERROR_SUCCESS)
        return 0;
    return WideCharToMultiByte(CP_UTF8, 0, w, -1, out, cap, NULL, NULL) > 1;
}

void os_describe(char *out, int cap) {
    /* GetVersionEx reports what the manifest allows: ask ntdll for the real version */
    typedef LONG(WINAPI * RtlGetVersionFn)(OSVERSIONINFOW *);
    OSVERSIONINFOW v;
    memset(&v, 0, sizeof(v));
    v.dwOSVersionInfoSize = sizeof(v);
    RtlGetVersionFn rgv = (RtlGetVersionFn)(void *)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion");
    if (rgv) rgv(&v);
    char product[128] = "Windows", display[32] = "";
    reg_str(L"ProductName", product, sizeof(product));
    reg_str(L"DisplayVersion", display, sizeof(display));
    /* Windows 11 still says "Windows 10" in ProductName */
    char *ten = strstr(product, "Windows 10");
    if (ten && v.dwBuildNumber >= 22000) ten[9] = '1';
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    const char *arch = si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64 ? "x64"
                     : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64 ? "arm64"
                     : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL ? "x86" : "unknown";
    snprintf(out, (size_t)cap, "%s%s%s (version %lu.%lu, build %lu), %s", product, display[0] ? " " : "", display,
             (unsigned long)v.dwMajorVersion, (unsigned long)v.dwMinorVersion, (unsigned long)v.dwBuildNumber, arch);
}
#else
#include <sys/utsname.h>
void os_describe(char *out, int cap) {
    struct utsname u;
    if (uname(&u) != 0) { snprintf(out, (size_t)cap, "Unix"); return; }
    char pretty[288] = "";
    FILE *f = fopen("/etc/os-release", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (!strncmp(line, "PRETTY_NAME=", 12)) {
                char *s = line + 12, *e;
                if (*s == '"') s++;
                if ((e = strpbrk(s, "\"\n"))) *e = 0;
                snprintf(pretty, sizeof(pretty), "%s", s);
                break;
            }
        fclose(f);
    }
#if defined(__APPLE__)
    char ver[32] = "";
    size_t vl = sizeof(ver);
    if (sysctlbyname("kern.osproductversion", ver, &vl, NULL, 0) == 0) snprintf(pretty, sizeof(pretty), "macOS %s", ver);
#endif
    if (pretty[0]) snprintf(out, (size_t)cap, "%s (%s %s), %s", pretty, u.sysname, u.release, u.machine);
    else snprintf(out, (size_t)cap, "%s %s, %s", u.sysname, u.release, u.machine);
}
#endif

/* ------------------------------------------------------------------ system info */

#if defined(PITHOS_X86)
static void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
    int t[4];
    __cpuidex(t, leaf, sub);
    for (int i = 0; i < 4; i++) r[i] = (unsigned)t[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static uint64_t xgetbv0(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif

void sys_info(SysInfo *si) {
    memset(si, 0, sizeof(*si));
#if defined(PITHOS_X86)
    unsigned r[4];
    cpuid(0, 0, r);
    unsigned max_leaf = r[0];
    cpuid(1, 0, r);
    int osxsave = (r[2] >> 27) & 1;
    int avx = (r[2] >> 28) & 1;
    si->has_fma = (r[2] >> 12) & 1;
    si->has_f16c = (r[2] >> 29) & 1;
    uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    int os_avx = (xcr0 & 6) == 6;
    int os_avx512 = (xcr0 & 0xe6) == 0xe6;
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        si->has_avx2 = avx && os_avx && ((r[1] >> 5) & 1);
        si->has_avx512f = os_avx512 && ((r[1] >> 16) & 1);
        si->has_avx512bw = os_avx512 && ((r[1] >> 30) & 1);
        si->has_avx512vnni = os_avx512 && ((r[2] >> 11) & 1);
        cpuid(7, 1, r);
        si->has_avxvnni = os_avx && ((r[0] >> 4) & 1);
    }
    if (!os_avx) si->has_fma = si->has_f16c = 0;
    cpuid(0x80000000, 0, r);
    if (r[0] >= 0x80000004) {
        unsigned b[12];
        cpuid(0x80000002, 0, b);
        cpuid(0x80000003, 0, b + 4);
        cpuid(0x80000004, 0, b + 8);
        memcpy(si->brand, b, 48);
        si->brand[48] = 0;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    si->has_neon = 1;
#if defined(__ARM_FEATURE_DOTPROD)
    si->has_dotprod = 1;
#endif
    strcpy(si->brand, "ARM64");
#endif

#if defined(_WIN32)
    SYSTEM_INFO s;
    GetSystemInfo(&s);
    si->n_logical = (int)s.dwNumberOfProcessors;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len) {
        uint8_t *buf = (uint8_t *)malloc(len);
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                       (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
            int n = 0;
            for (DWORD off = 0; off < len;) {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
                n++;
                off += e->Size;
            }
            si->n_physical = n;
        }
        free(buf);
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        si->total_ram = ms.ullTotalPhys;
        si->avail_ram = ms.ullAvailPhys;
    }
#else
    si->n_logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
#if defined(__APPLE__)
    int v = 0; size_t vl = sizeof(v);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &vl, NULL, 0) == 0 && v > 0) si->n_physical = v;
    else if (sysctlbyname("hw.physicalcpu", &v, &vl, NULL, 0) == 0) si->n_physical = v;
    uint64_t mem = 0; size_t ml = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &ml, NULL, 0) == 0) si->total_ram = mem;
    si->avail_ram = si->total_ram / 2;
#else
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        int cores = 0, sockets_seen = 0;
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "cpu cores", 9) && !sockets_seen) {
                char *c = strchr(line, ':');
                if (c) cores = atoi(c + 1);
                sockets_seen = 1;
            }
        }
        fclose(f);
        if (cores > 0) si->n_physical = cores;
    }
    f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        unsigned long long v;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %llu kB", &v) == 1) si->total_ram = v * 1024ull;
            if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) si->avail_ram = v * 1024ull;
        }
        fclose(f);
    }
#endif
#endif
    if (si->n_logical <= 0) si->n_logical = 1;
    if (si->n_physical <= 0 || si->n_physical > si->n_logical) si->n_physical = si->n_logical;
}

/* ------------------------------------------------------------------ thread pool */

#if defined(_WIN32)
typedef HANDLE thread_t;
typedef SRWLOCK mutex_t;
typedef CONDITION_VARIABLE cond_t;
#define mutex_init(m) InitializeSRWLock(m)
#define mutex_lock(m) AcquireSRWLockExclusive(m)
#define mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define cond_init(c) InitializeConditionVariable(c)
#define cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, 0)
#define cond_broadcast(c) WakeAllConditionVariable(c)
#else
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t cond_t;
#define mutex_init(m) pthread_mutex_init(m, NULL)
#define mutex_lock(m) pthread_mutex_lock(m)
#define mutex_unlock(m) pthread_mutex_unlock(m)
#define cond_init(c) pthread_cond_init(c, NULL)
#define cond_wait(c, m) pthread_cond_wait(c, m)
#define cond_broadcast(c) pthread_cond_broadcast(c)
#endif

/* Spin this many pause iterations (~5-10 ms) before sleeping on the condition variable. */
#define SPIN_ITERS 200000

#if defined(_MSC_VER) && !defined(__clang__)
typedef volatile __int64 atomic_i64;
static __forceinline int64_t a64_load(atomic_i64 *p) { int64_t v = *p; _ReadWriteBarrier(); return v; }
static __forceinline void a64_store(atomic_i64 *p, int64_t v) { _InterlockedExchange64(p, v); }
static __forceinline int a64_cas(atomic_i64 *p, int64_t expect, int64_t want) {
    return _InterlockedCompareExchange64(p, want, expect) == expect;
}
static __forceinline void full_fence(void) { MemoryBarrier(); }
#else
typedef volatile int64_t atomic_i64;
static inline int64_t a64_load(atomic_i64 *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void a64_store(atomic_i64 *p, int64_t v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
static inline int a64_cas(atomic_i64 *p, int64_t expect, int64_t want) {
    return __atomic_compare_exchange_n(p, &expect, want, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
static inline void full_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
#endif

typedef struct {
    ThreadPool *pool;
    int id;
} WorkerArg;

typedef struct {
    task_fn fn;
    void *ctx;
    int n_tasks;
} PoolJob;

/* Jobs are double-buffered by generation parity and the shared task counter carries the
 * generation in its high 32 bits. A worker that wakes up late (or was preempted by the OS)
 * can neither delay completion nor claim a task of a newer job: its compare-and-swap fails.
 * Completion is counted in tasks, so the caller never waits for idle workers. */
struct ThreadPool {
    int n;                  /* total threads including caller */
    thread_t *threads;
    WorkerArg *args;
    mutex_t mtx;
    cond_t cv;
    PoolJob jobs[2];
    PITHOS_ALIGN(64) atomic_i32 gen;
    PITHOS_ALIGN(64) atomic_i64 next;     /* (gen << 32) | next task index */
    PITHOS_ALIGN(64) atomic_i32 done;     /* tasks finished in the current job */
    PITHOS_ALIGN(64) atomic_i32 sleepers;
    int stop;
};

static void run_tasks(ThreadPool *p, uint32_t g, int tid) {
    const PoolJob *j = &p->jobs[g & 1];
    task_fn fn = j->fn;
    void *ctx = j->ctx;
    const int n = j->n_tasks;
    for (;;) {
        int64_t v = a64_load(&p->next);
        if ((uint32_t)((uint64_t)v >> 32) != g) return;
        int idx = (int)(uint32_t)v;
        if (idx >= n) return;
        if (!a64_cas(&p->next, v, v + 1)) continue;
        fn(ctx, idx, tid);
        atomic_fetch_add_i32(&p->done, 1);
    }
}

static void worker_loop(WorkerArg *a) {
    ThreadPool *p = a->pool;
    int seen = atomic_load_i32(&p->gen);
    for (;;) {
        int g, spins = 0;
        while ((g = atomic_load_i32(&p->gen)) == seen) {
            if (++spins < SPIN_ITERS) { cpu_relax(); continue; }
            mutex_lock(&p->mtx);
            atomic_fetch_add_i32(&p->sleepers, 1);
            while ((g = atomic_load_i32(&p->gen)) == seen) cond_wait(&p->cv, &p->mtx);
            atomic_fetch_add_i32(&p->sleepers, -1);
            mutex_unlock(&p->mtx);
            break;
        }
        seen = g;
        if (p->stop) return;
        run_tasks(p, (uint32_t)g, a->id);
    }
}

#if defined(_WIN32)
static DWORD WINAPI worker_entry(LPVOID arg) { worker_loop((WorkerArg *)arg); return 0; }
#else
static void *worker_entry(void *arg) { worker_loop((WorkerArg *)arg); return NULL; }
#endif

ThreadPool *pool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    ThreadPool *p = (ThreadPool *)aligned_malloc(sizeof(ThreadPool), 64);
    memset(p, 0, sizeof(*p));
    p->n = n_threads;
    mutex_init(&p->mtx);
    cond_init(&p->cv);
    p->threads = (thread_t *)calloc((size_t)n_threads, sizeof(thread_t));
    p->args = (WorkerArg *)calloc((size_t)n_threads, sizeof(WorkerArg));
    for (int i = 1; i < n_threads; i++) {
        p->args[i].pool = p;
        p->args[i].id = i;
#if defined(_WIN32)
        p->threads[i] = CreateThread(NULL, 1 << 20, worker_entry, &p->args[i], 0, NULL);
#else
        pthread_create(&p->threads[i], NULL, worker_entry, &p->args[i]);
#endif
    }
    return p;
}

static void publish(ThreadPool *p, int g) {
    atomic_store_i32(&p->gen, g);
    full_fence();
    if (atomic_load_i32(&p->sleepers) > 0) {
        mutex_lock(&p->mtx);
        cond_broadcast(&p->cv);
        mutex_unlock(&p->mtx);
    }
}

void pool_destroy(ThreadPool *p) {
    if (!p) return;
    p->stop = 1;
    mutex_lock(&p->mtx);
    atomic_fetch_add_i32(&p->gen, 1);
    cond_broadcast(&p->cv);
    mutex_unlock(&p->mtx);
    for (int i = 1; i < p->n; i++) {
#if defined(_WIN32)
        WaitForSingleObject(p->threads[i], INFINITE);
        CloseHandle(p->threads[i]);
#else
        pthread_join(p->threads[i], NULL);
#endif
    }
    free(p->threads);
    free(p->args);
    aligned_free(p);
}

int pool_size(const ThreadPool *p) { return p->n; }

void pool_run(ThreadPool *p, task_fn fn, void *ctx, int n_tasks) {
    if (n_tasks <= 0) return;
    if (p->n == 1 || n_tasks == 1) {
        for (int i = 0; i < n_tasks; i++) fn(ctx, i, 0);
        return;
    }
    const int g = atomic_load_i32(&p->gen) + 1;
    PoolJob *j = &p->jobs[g & 1];
    j->fn = fn;
    j->ctx = ctx;
    j->n_tasks = n_tasks;
    atomic_store_i32(&p->done, 0);
    a64_store(&p->next, (int64_t)((uint64_t)(uint32_t)g << 32));
    publish(p, g);
    run_tasks(p, (uint32_t)g, 0);
    while (atomic_load_i32(&p->done) < n_tasks) cpu_relax();
}

/* ------------------------------------------------------------------ background prefetcher */

#define PF_QUEUE 256

struct Prefetcher {
    thread_t th;
    mutex_t mtx;
    cond_t cv;
    const void *addr[PF_QUEUE];
    size_t len[PF_QUEUE];
    int head, count, stop;
};

static void prefetcher_loop(Prefetcher *p) {
    for (;;) {
        mutex_lock(&p->mtx);
        while (!p->count && !p->stop) cond_wait(&p->cv, &p->mtx);
        if (p->stop) { mutex_unlock(&p->mtx); return; }
        const void *a = p->addr[p->head];
        size_t l = p->len[p->head];
        p->head = (p->head + 1) % PF_QUEUE;
        p->count--;
        mutex_unlock(&p->mtx);
        prefetch_memory(a, l);
    }
}

#if defined(_WIN32)
static DWORD WINAPI prefetcher_entry(LPVOID arg) { prefetcher_loop((Prefetcher *)arg); return 0; }
#else
static void *prefetcher_entry(void *arg) { prefetcher_loop((Prefetcher *)arg); return NULL; }
#endif

Prefetcher *prefetcher_create(void) {
    Prefetcher *p = (Prefetcher *)calloc(1, sizeof(Prefetcher));
    mutex_init(&p->mtx);
    cond_init(&p->cv);
#if defined(_WIN32)
    p->th = CreateThread(NULL, 1 << 16, prefetcher_entry, p, 0, NULL);
#else
    pthread_create(&p->th, NULL, prefetcher_entry, p);
#endif
    return p;
}

void prefetcher_push(Prefetcher *p, const void *addr, size_t len) {
    if (!p || !len) return;
    mutex_lock(&p->mtx);
    if (p->count < PF_QUEUE) {
        int tail = (p->head + p->count) % PF_QUEUE;
        p->addr[tail] = addr;
        p->len[tail] = len;
        p->count++;
        cond_broadcast(&p->cv);
    }
    mutex_unlock(&p->mtx);
}

void prefetcher_clear(Prefetcher *p) {
    if (!p) return;
    mutex_lock(&p->mtx);
    p->count = 0;
    mutex_unlock(&p->mtx);
}

void prefetcher_destroy(Prefetcher *p) {
    if (!p) return;
    mutex_lock(&p->mtx);
    p->stop = 1;
    cond_broadcast(&p->cv);
    mutex_unlock(&p->mtx);
#if defined(_WIN32)
    WaitForSingleObject(p->th, INFINITE);
    CloseHandle(p->th);
#else
    pthread_join(p->th, NULL);
#endif
    free(p);
}

/* ------------------------------------------------------------------ generic thread / lock */

struct PithosThread {
    thread_t th;
    void (*fn)(void *);
    void *arg;
};

#if defined(_WIN32)
static DWORD WINAPI generic_entry(LPVOID p) { PithosThread *t = (PithosThread *)p; t->fn(t->arg); return 0; }
#else
static void *generic_entry(void *p) { PithosThread *t = (PithosThread *)p; t->fn(t->arg); return NULL; }
#endif

PithosThread *thread_create(void (*fn)(void *), void *arg) {
    PithosThread *t = (PithosThread *)calloc(1, sizeof(PithosThread));
    t->fn = fn;
    t->arg = arg;
#if defined(_WIN32)
    t->th = CreateThread(NULL, 1 << 20, generic_entry, t, 0, NULL);
#else
    pthread_create(&t->th, NULL, generic_entry, t);
#endif
    return t;
}

void thread_join(PithosThread *t) {
    if (!t) return;
#if defined(_WIN32)
    WaitForSingleObject(t->th, INFINITE);
    CloseHandle(t->th);
#else
    pthread_join(t->th, NULL);
#endif
    free(t);
}

struct PithosLock { mutex_t m; cond_t c; };

PithosLock *lock_create(void) {
    PithosLock *l = (PithosLock *)calloc(1, sizeof(PithosLock));
    mutex_init(&l->m);
    cond_init(&l->c);
    return l;
}
void lock_destroy(PithosLock *l) { free(l); }
void lock_acquire(PithosLock *l) { mutex_lock(&l->m); }
void lock_release(PithosLock *l) { mutex_unlock(&l->m); }
void lock_wait(PithosLock *l) { cond_wait(&l->c, &l->m); }
void lock_notify_all(PithosLock *l) { cond_broadcast(&l->c); }

/* ------------------------------------------------------------------ dynamic libraries */

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

void *lib_open(const char *name) {
#if defined(_WIN32)
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *lib_sym(void *lib, const char *name) {
    if (!lib) return NULL;
#if defined(_WIN32)
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

/* ------------------------------------------------------------------ detached threads */

typedef struct { void (*fn)(void *); void *arg; } SpawnArg;

#if defined(_WIN32)
static DWORD WINAPI spawn_entry(LPVOID p) {
#else
static void *spawn_entry(void *p) {
#endif
    SpawnArg a = *(SpawnArg *)p;
    free(p);
    a.fn(a.arg);
    return 0;
}

int thread_spawn(void (*fn)(void *), void *arg) {
    SpawnArg *a = (SpawnArg *)malloc(sizeof(SpawnArg));
    a->fn = fn;
    a->arg = arg;
#if defined(_WIN32)
    HANDLE h = CreateThread(NULL, 1 << 20, spawn_entry, a, 0, NULL);
    if (!h) { free(a); return 0; }
    CloseHandle(h);
#else
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&at, 1 << 20);
    int rc = pthread_create(&th, &at, spawn_entry, a);
    pthread_attr_destroy(&at);
    if (rc) { free(a); return 0; }
#endif
    return 1;
}

/* ------------------------------------------------------------------ files and directories (UTF-8 paths) */

#if defined(_WIN32)
static int utf8_to_wide(const char *s, wchar_t *w, int cap) {
    return MultiByteToWideChar(CP_UTF8, 0, s, -1, w, cap) > 0;
}
#else
#include <dirent.h>
#include <errno.h>
#endif

FILE *file_open(const char *path, const char *mode) {
#if defined(_WIN32)
    wchar_t wp[1024], wm[16];
    if (!utf8_to_wide(path, wp, 1024) || !utf8_to_wide(mode, wm, 16)) return NULL;
    return _wfopen(wp, wm);
#else
    return fopen(path, mode);
#endif
}

int file_remove(const char *path) {
#if defined(_WIN32)
    wchar_t wp[1024];
    return utf8_to_wide(path, wp, 1024) && DeleteFileW(wp);
#else
    return unlink(path) == 0;
#endif
}

int file_rename(const char *from, const char *to) {
#if defined(_WIN32)
    wchar_t wf[1024], wt[1024];
    if (!utf8_to_wide(from, wf, 1024) || !utf8_to_wide(to, wt, 1024)) return 0;
    return MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(from, to) == 0;
#endif
}

int dir_create(const char *path) {
#if defined(_WIN32)
    wchar_t wp[1024];
    if (!utf8_to_wide(path, wp, 1024)) return 0;
    return CreateDirectoryW(wp, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(path, 0755) == 0 || errno == EEXIST;
#endif
}

int dir_list(const char *dir, void (*cb)(const char *name, void *u), void *u) {
#if defined(_WIN32)
    char pat[1100];
    wchar_t wp[1100];
    snprintf(pat, sizeof(pat), "%s/*", dir);
    if (!utf8_to_wide(pat, wp, 1100)) return 0;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wp, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char name[1024];
        if (WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name, sizeof(name), NULL, NULL) > 0) cb(name, u);
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return 1;
#else
    DIR *d = opendir(dir);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL)
        if (e->d_name[0] != '.') cb(e->d_name, u);
    closedir(d);
    return 1;
#endif
}

int lock_wait_ms(PithosLock *l, int ms) {
#if defined(_WIN32)
    return SleepConditionVariableSRW(&l->c, &l->m, (DWORD)ms, 0) ? 1 : 0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(&l->c, &l->m, &ts) == 0;
#endif
}

int dir_entries(const char *dir, void (*cb)(const char *name, int is_dir, uint64_t size, void *u), void *u) {
#if defined(_WIN32)
    char pat[1100];
    wchar_t wp[1100];
    snprintf(pat, sizeof(pat), "%s/*", dir);
    if (!utf8_to_wide(pat, wp, 1100)) return 0;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wp, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
        if (fd.dwFileAttributes & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM)) continue;
        char name[1024];
        if (WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name, sizeof(name), NULL, NULL) > 0)
            cb(name, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0,
               ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow, u);
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return 1;
#else
    DIR *d = opendir(dir);
    if (!d) return 0;
    struct dirent *e;
    char path[2048];
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        struct stat st;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (stat(path, &st)) continue;
        cb(e->d_name, S_ISDIR(st.st_mode), (uint64_t)st.st_size, u);
    }
    closedir(d);
    return 1;
#endif
}
