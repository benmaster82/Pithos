/* proc.h - child processes with pipes (used for MCP servers and for curl). */
#ifndef PITHOS_PROC_H
#define PITHOS_PROC_H

typedef struct Proc Proc;

/* Start argv[0] (searched in PATH; on Windows also .cmd/.bat scripts such as npx) with
 * stdin/stdout connected to pipes. env: NULL-terminated "KEY=VALUE" entries added to the
 * inherited environment, or NULL. quiet: discard stderr instead of sharing ours.
 * Children and their descendants are terminated together with this process (Windows) or by
 * proc_kill / proc_free (POSIX: the child gets its own process group). */
Proc *proc_spawn(const char *const *argv, const char *const *env, int quiet, char *err, int errlen);
int   proc_write(Proc *p, const char *s, int n);   /* 1 if everything was written */
int   proc_read(Proc *p, char *buf, int cap);      /* blocking: bytes read, 0 at end of output */
void  proc_close_stdin(Proc *p);
int   proc_wait(Proc *p, int timeout_ms);          /* exit code, or -1 if still running */
void  proc_kill(Proc *p);                          /* the child and everything it started */
/* Close stdin, give the child grace_ms to exit by itself, then kill it with its descendants:
 * afterwards proc_read returns 0, so a reader thread can be joined before proc_free. */
void  proc_stop(Proc *p, int grace_ms);
void  proc_free(Proc *p);                          /* kills the child if it is still running */

/* Run to completion capturing stdout (at most max_bytes). *out is malloc'ed and NUL-terminated.
 * Returns the exit code, -1 if the program cannot be started, -2 on timeout. */
int proc_run(const char *const *argv, char **out, int *out_len, int max_bytes, int timeout_ms);

#endif
