/* proc.c - see proc.h. Win32: CreateProcessW + anonymous pipes + a kill-on-close job object.
 * POSIX: fork/execvp (children see end of input and stop when this process ends). */
#include "proc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "platform.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

struct Proc {
    HANDLE h, in, out, err;
    HANDLE job; /* the child and everything it starts (npx -> node): killed together */
    DWORD pid;
};

typedef struct { wchar_t *s; size_t n, cap; } WBuf;

static void wb_add(WBuf *b, const wchar_t *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2 + 64;
        b->s = (wchar_t *)realloc(b->s, b->cap * sizeof(wchar_t));
    }
    memcpy(b->s + b->n, s, n * sizeof(wchar_t));
    b->n += n;
    b->s[b->n] = 0;
}

static wchar_t *widen(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)(n > 0 ? n : 1));
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    else w[0] = 0;
    return w;
}

/* Quote one argument following the rules of CommandLineToArgvW / the MSVC runtime. */
static void quote_arg(WBuf *b, const wchar_t *a) {
    if (*a && !wcspbrk(a, L" \t\n\v\"")) { wb_add(b, a, wcslen(a)); return; }
    wb_add(b, L"\"", 1);
    for (const wchar_t *p = a;; p++) {
        size_t bs = 0;
        while (*p == L'\\') { p++; bs++; }
        if (!*p) {
            for (size_t i = 0; i < bs * 2; i++) wb_add(b, L"\\", 1);
            break;
        }
        if (*p == L'"') {
            for (size_t i = 0; i < bs * 2 + 1; i++) wb_add(b, L"\\", 1);
        } else {
            for (size_t i = 0; i < bs; i++) wb_add(b, L"\\", 1);
        }
        wb_add(b, p, 1);
    }
    wb_add(b, L"\"", 1);
}

/* One job per child: closing its handle (explicitly, or when this process ends) kills the
 * child and all of its descendants. */
static HANDLE kill_job(void) {
    HANDLE job = CreateJobObjectW(NULL, NULL);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li;
        memset(&li, 0, sizeof(li));
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    }
    return job;
}

Proc *proc_start(const ProcOptions *o, char *err, int errlen) {
    const char *const *argv = o->argv, *const *env = o->env;
    if (err && errlen) err[0] = 0;
    wchar_t *prog = widen(argv[0]);
    /* resolve the program: PATH search with .exe, then .cmd / .bat (npx, uvx shims) */
    wchar_t path[MAX_PATH * 2];
    const wchar_t *exts[] = { L".exe", L".cmd", L".bat", L".com" };
    int found = 0;
    for (int i = 0; i < 4 && !found; i++)
        found = SearchPathW(NULL, prog, exts[i], MAX_PATH * 2, path, NULL) > 0;
    if (!found) {
        snprintf(err, (size_t)errlen, "program not found: %s", argv[0]);
        free(prog);
        return NULL;
    }
    free(prog);
    size_t pl = wcslen(path);
    int script = pl > 4 && (!_wcsicmp(path + pl - 4, L".cmd") || !_wcsicmp(path + pl - 4, L".bat"));

    WBuf cmd = { 0 };
    if (o->win_cmdline) {
        wchar_t *w = widen(o->win_cmdline);
        wb_add(&cmd, w, wcslen(w));
        free(w);
        script = 0;
    } else {
        if (script) wb_add(&cmd, L"cmd.exe /d /s /c \"", 18);
        quote_arg(&cmd, path);
        for (int i = 1; argv[i]; i++) {
            wchar_t *w = widen(argv[i]);
            wb_add(&cmd, L" ", 1);
            quote_arg(&cmd, w);
            free(w);
        }
        if (script) wb_add(&cmd, L"\"", 1);
    }

    WBuf envb = { 0 };
    if (env && env[0]) {
        wchar_t *cur = GetEnvironmentStringsW();
        size_t n = 0;
        while (cur[n] || cur[n + 1]) n++;
        wb_add(&envb, cur, n + 1);
        FreeEnvironmentStringsW(cur);
        for (int i = 0; env[i]; i++) {
            wchar_t *w = widen(env[i]);
            wb_add(&envb, w, wcslen(w) + 1);
            free(w);
        }
        wb_add(&envb, L"\0", 1);
    }

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE in_r, in_w, out_r, out_w, errh = NULL, err_r = NULL;
    if (!CreatePipe(&in_r, &in_w, &sa, 0)) { free(cmd.s); free(envb.s); return NULL; }
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
        CloseHandle(in_r); CloseHandle(in_w); free(cmd.s); free(envb.s);
        return NULL;
    }
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    if (o->err_mode == PROC_ERR_PIPE) {
        if (!CreatePipe(&err_r, &errh, &sa, 0)) err_r = errh = NULL;
        else SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);
    } else if (o->err_mode == PROC_ERR_NULL) {
        errh = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    } else {
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &errh, 0, TRUE,
                        DUPLICATE_SAME_ACCESS);
    }
    wchar_t *wcwd = o->cwd && *o->cwd ? widen(o->cwd) : NULL;

    STARTUPINFOW si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = errh;
    PROCESS_INFORMATION pi;
    BOOL ok = CreateProcessW(script ? NULL : path, cmd.s, NULL, NULL, TRUE,
                             CREATE_NO_WINDOW | CREATE_SUSPENDED | (envb.s ? CREATE_UNICODE_ENVIRONMENT : 0),
                             envb.s, wcwd, &si, &pi);
    DWORD gle = GetLastError();
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (errh && errh != INVALID_HANDLE_VALUE) CloseHandle(errh);
    free(cmd.s);
    free(envb.s);
    free(wcwd);
    if (!ok) {
        if (gle == ERROR_DIRECTORY || (o->cwd && gle == ERROR_PATH_NOT_FOUND))
            snprintf(err, (size_t)errlen, "folder not found: %s", o->cwd);
        else
            snprintf(err, (size_t)errlen, "cannot start %s (error %lu)", argv[0], (unsigned long)gle);
        CloseHandle(in_w);
        CloseHandle(out_r);
        if (err_r) CloseHandle(err_r);
        return NULL;
    }
    HANDLE job = kill_job();
    if (job && !AssignProcessToJobObject(job, pi.hProcess)) { CloseHandle(job); job = NULL; }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    Proc *p = (Proc *)calloc(1, sizeof(Proc));
    p->h = pi.hProcess;
    p->in = in_w;
    p->out = out_r;
    p->err = err_r;
    p->job = job;
    p->pid = pi.dwProcessId;
    return p;
}

int proc_pid(Proc *p) { return (int)p->pid; }

int proc_read_err(Proc *p, char *buf, int cap) {
    DWORD r = 0;
    if (!p->err || !ReadFile(p->err, buf, (DWORD)cap, &r, NULL)) return 0;
    return (int)r;
}

int proc_write(Proc *p, const char *s, int n) {
    while (n > 0 && p->in) {
        DWORD w = 0;
        if (!WriteFile(p->in, s, (DWORD)n, &w, NULL) || w == 0) return 0;
        s += w;
        n -= (int)w;
    }
    return n == 0;
}

int proc_read(Proc *p, char *buf, int cap) {
    DWORD r = 0;
    if (!ReadFile(p->out, buf, (DWORD)cap, &r, NULL)) return 0;
    return (int)r;
}

void proc_close_stdin(Proc *p) {
    if (p->in) { CloseHandle(p->in); p->in = NULL; }
}

int proc_wait(Proc *p, int timeout_ms) {
    if (WaitForSingleObject(p->h, timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms) != WAIT_OBJECT_0) return -1;
    DWORD code = 0;
    GetExitCodeProcess(p->h, &code);
    return (int)code;
}

void proc_kill(Proc *p) {
    if (p->job) TerminateJobObject(p->job, 1);
    else TerminateProcess(p->h, 1);
}

void proc_free(Proc *p) {
    if (!p) return;
    proc_kill(p); /* also descendants that outlived the child */
    proc_close_stdin(p);
    CloseHandle(p->out);
    if (p->err) CloseHandle(p->err);
    CloseHandle(p->h);
    if (p->job) CloseHandle(p->job);
    free(p);
}

#else /* POSIX */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

struct Proc {
    pid_t pid;
    int in, out, err, done, code;
};

Proc *proc_start(const ProcOptions *o, char *err, int errlen) {
    const char *const *argv = o->argv, *const *env = o->env;
    if (err && errlen) err[0] = 0;
    int pin[2], pout[2], pexec[2], perr[2] = { -1, -1 };
    if (pipe(pin) || pipe(pout) || pipe(pexec) || (o->err_mode == PROC_ERR_PIPE && pipe(perr))) {
        snprintf(err, (size_t)errlen, "pipe failed");
        return NULL;
    }
    fcntl(pexec[1], F_SETFD, FD_CLOEXEC);
    pid_t pid = fork();
    if (pid < 0) { snprintf(err, (size_t)errlen, "fork failed"); return NULL; }
    if (pid == 0) {
        /* no PR_SET_PDEATHSIG: it fires when the spawning *thread* exits. MCP servers stop
         * by themselves when their stdin closes, i.e. when this process ends. Own process
         * group, so that proc_kill also reaches what the child starts (npx -> node). */
        setpgid(0, 0);
        dup2(pin[0], 0);
        dup2(pout[1], 1);
        if (o->err_mode == PROC_ERR_PIPE) {
            dup2(perr[1], 2);
            close(perr[0]); close(perr[1]);
        } else if (o->err_mode == PROC_ERR_NULL) {
            int nul = open("/dev/null", O_WRONLY);
            if (nul >= 0) dup2(nul, 2);
        }
        close(pin[0]); close(pin[1]); close(pout[0]); close(pout[1]); close(pexec[0]);
        int e = 0, stage = 0; /* stage 1: chdir failed */
        if (o->cwd && *o->cwd && chdir(o->cwd) != 0) { e = errno; stage = 1; }
        else {
            for (int i = 0; env && env[i]; i++) putenv((char *)env[i]);
            execvp(argv[0], (char *const *)argv);
            e = errno;
        }
        int msg[2] = { stage, e };
        if (write(pexec[1], msg, sizeof(msg)) < 0) {}
        _exit(127);
    }
    close(pin[0]);
    close(pout[1]);
    close(pexec[1]);
    if (perr[1] >= 0) close(perr[1]);
    int msg[2];
    if (read(pexec[0], msg, sizeof(msg)) == (ssize_t)sizeof(msg)) { /* chdir or exec failed */
        close(pexec[0]);
        close(pin[1]);
        close(pout[0]);
        if (perr[0] >= 0) close(perr[0]);
        waitpid(pid, NULL, 0);
        if (msg[0] == 1) snprintf(err, (size_t)errlen, "folder not found: %s", o->cwd);
        else snprintf(err, (size_t)errlen, "cannot start %s: %s", argv[0], strerror(msg[1]));
        return NULL;
    }
    close(pexec[0]);
    Proc *p = (Proc *)calloc(1, sizeof(Proc));
    p->pid = pid;
    p->in = pin[1];
    p->out = pout[0];
    p->err = perr[0];
    return p;
}

int proc_pid(Proc *p) { return (int)p->pid; }

int proc_read_err(Proc *p, char *buf, int cap) {
    if (p->err < 0) return 0;
    for (;;) {
        ssize_t r = read(p->err, buf, (size_t)cap);
        if (r < 0 && errno == EINTR) continue;
        return r > 0 ? (int)r : 0;
    }
}

int proc_write(Proc *p, const char *s, int n) {
    while (n > 0 && p->in >= 0) {
        ssize_t w = write(p->in, s, (size_t)n);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return 0; }
        s += w;
        n -= (int)w;
    }
    return n == 0;
}

int proc_read(Proc *p, char *buf, int cap) {
    for (;;) {
        ssize_t r = read(p->out, buf, (size_t)cap);
        if (r < 0 && errno == EINTR) continue;
        return r > 0 ? (int)r : 0;
    }
}

void proc_close_stdin(Proc *p) {
    if (p->in >= 0) { close(p->in); p->in = -1; }
}

int proc_wait(Proc *p, int timeout_ms) {
    if (p->done) return p->code;
    for (int waited = 0;; waited += 10) {
        int st;
        pid_t r = waitpid(p->pid, &st, WNOHANG);
        if (r == p->pid) {
            p->done = 1;
            p->code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            return p->code;
        }
        if (timeout_ms >= 0 && waited >= timeout_ms) return -1;
        struct timespec ts = { 0, 10 * 1000000L };
        nanosleep(&ts, NULL);
    }
}

void proc_kill(Proc *p) {
    kill(-p->pid, SIGKILL); /* the whole group: descendants may outlive the child */
    if (!p->done) kill(p->pid, SIGKILL);
}

void proc_free(Proc *p) {
    if (!p) return;
    if (!p->done) { kill(-p->pid, SIGTERM); proc_wait(p, 500); }
    proc_kill(p);
    if (!p->done) proc_wait(p, -1);
    proc_close_stdin(p);
    close(p->out);
    if (p->err >= 0) close(p->err);
    free(p);
}
#endif

Proc *proc_spawn(const char *const *argv, const char *const *env, int quiet, char *err, int errlen) {
    ProcOptions o;
    memset(&o, 0, sizeof(o));
    o.argv = argv;
    o.env = env;
    o.err_mode = quiet ? PROC_ERR_NULL : PROC_ERR_SHARE;
    return proc_start(&o, err, errlen);
}

/* ------------------------------------------------------------------ stop */

void proc_stop(Proc *p, int grace_ms) {
    proc_close_stdin(p);
    proc_wait(p, grace_ms);
    proc_kill(p); /* descendants too, even when the child itself has left */
}

/* ------------------------------------------------------------------ run to completion */

typedef struct {
    Proc *p;
    char *buf;
    int n, cap, max;
} RunCtx;

static void run_reader(void *arg) {
    RunCtx *rc = (RunCtx *)arg;
    char tmp[16384];
    int k;
    while ((k = proc_read(rc->p, tmp, sizeof(tmp))) > 0) {
        int take = rc->n + k > rc->max ? rc->max - rc->n : k;
        if (take <= 0) continue; /* keep draining so the child is not blocked */
        if (rc->n + take + 1 > rc->cap) {
            rc->cap = (rc->n + take + 1) * 2;
            rc->buf = (char *)realloc(rc->buf, (size_t)rc->cap);
        }
        memcpy(rc->buf + rc->n, tmp, (size_t)take);
        rc->n += take;
    }
}

int proc_run(const char *const *argv, char **out, int *out_len, int max_bytes, int timeout_ms) {
    char err[256];
    *out = NULL;
    if (out_len) *out_len = 0;
    Proc *p = proc_spawn(argv, NULL, 1, err, sizeof(err));
    if (!p) return -1;
    proc_close_stdin(p);
    RunCtx rc = { p, NULL, 0, 0, max_bytes };
    PithosThread *t = thread_create(run_reader, &rc);
    int code = proc_wait(p, timeout_ms);
    if (code == -1) {
        proc_kill(p);
        code = -2;
    }
    thread_join(t);
    proc_free(p);
    if (!rc.buf) rc.buf = (char *)malloc(1);
    rc.buf[rc.n] = 0;
    *out = rc.buf;
    if (out_len) *out_len = rc.n;
    return code;
}
