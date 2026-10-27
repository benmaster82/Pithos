/* test_shell.c - the agent's shell tools without the model: run_command (exit codes, stderr,
 * timeout, working folder), background processes (start, read with wait/until, stop, list),
 * edit_file and the destructive-command check. Run from an empty scratch folder:
 *     test_shell            (exit code 0 when every check passes) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/json.h"
#include "../src/platform.h"
#include "../src/tools.h"

static int fails;

static char *call(Tools *t, const char *name, const char *args, int *ok) {
    Json *a = json_parse(args, (int)strlen(args), NULL, 0);
    Buf out = { 0 };
    double t0 = time_sec();
    int r = tools_call(t, name, a, &out);
    json_free(a);
    printf("\n> %s %s  [%s, %.2f s]\n%s\n", name, args, r ? "ok" : "error", time_sec() - t0, out.s ? out.s : "");
    if (ok) *ok = r;
    return out.s;
}

static void check(int cond, const char *what) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

int main(void) {
    ToolsConfig tc;
    memset(&tc, 0, sizeof(tc));
    tc.workspace = ".";
    tc.allow_write = 1;
    tc.shell = 1;
    Tools *t = tools_create(&tc);
    printf("%s\n", tools_env_info(t));
    int ok;
    char *r;
#if defined(_WIN32)
    const char *echo2 = "{\"command\": \"Write-Output out1; Write-Error err1; exit 3\"}";
    const char *slow = "{\"command\": \"Write-Output start; Start-Sleep 3; Write-Output ready; Start-Sleep 30\", \"wait\": 1}";
    const char *sleep = "{\"command\": \"Start-Sleep 20\", \"timeout\": 1}";
    const char *quick = "{\"command\": \"Write-Output quick-done\"}";
    const char *pwd = "{\"command\": \"(Get-Location).Path\", \"cwd\": \"sub\"}";
#else
    const char *echo2 = "{\"command\": \"echo out1; echo err1 >&2; exit 3\"}";
    const char *slow = "{\"command\": \"echo start; sleep 3; echo ready; sleep 30\", \"wait\": 1}";
    const char *sleep = "{\"command\": \"sleep 20\", \"timeout\": 1}";
    const char *quick = "{\"command\": \"echo quick-done\"}";
    const char *pwd = "{\"command\": \"pwd\", \"cwd\": \"sub\"}";
#endif
    r = call(t, "run_command", echo2, &ok);
    check(!ok && strstr(r, "exit code 3") && strstr(r, "out1") && strstr(r, "--- stderr ---") && strstr(r, "err1"),
          "exit code, stdout and stderr apart");
    free(r);
#if defined(_WIN32)
    r = call(t, "run_command", "{\"command\": \"Write-Output one && Write-Output 'two && three'\"}", &ok);
    check(ok && strstr(r, "one") && strstr(r, "two && three"), "'&&' works in Windows PowerShell 5.1");
    free(r);
    r = call(t, "run_command", "{\"command\": \"cmd /c exit 4 && Write-Output never\"}", &ok);
    check(!ok && strstr(r, "exit code 4") && !strstr(r, "\nnever"), "'&&' stops at the first failure");
    free(r);
    r = call(t, "run_command", "{\"command\": \"cmd /c exit 4 || Write-Output fallback\"}", &ok);
    check(ok && strstr(r, "fallback"), "'||' runs the fallback");
    free(r);
    r = call(t, "run_command", "{\"command\": \"Write-Output (1 +\"}", &ok);
    check(!ok && strstr(r, "--- stderr ---") && !strstr(r, " ? "), "parse errors readable (OEM code page decoded)");
    free(r);
#endif
    r = call(t, "run_command", sleep, &ok);
    check(!ok && strstr(r, "TIMED OUT"), "timeout stops the command");
    free(r);
    dir_create("sub");
    r = call(t, "run_command", pwd, &ok);
    check(ok && strstr(r, "sub"), "cwd");
    free(r);

    r = call(t, "start_process", slow, &ok);
    check(ok && strstr(r, "is running") && strstr(r, "\nstart") && !strstr(r, "\nready"),
          "start_process returns at once with the first output");
    free(r);
    double t0 = time_sec();
    r = call(t, "read_process", "{\"id\": 1, \"wait\": 20, \"until\": \"ready\"}", &ok);
    check(ok && strstr(r, "ready") && time_sec() - t0 < 10, "read_process waits until the text appears");
    free(r);
    r = call(t, "read_process", "{\"id\": 1}", &ok);
    check(ok && strstr(r, "no new output"), "read_process returns only new output");
    free(r);
    r = call(t, "start_process", quick, &ok);
    check(ok && strstr(r, "exited with code 0") && strstr(r, "quick-done"), "a short command ends within the wait");
    free(r);
    r = call(t, "list_processes", "{}", &ok);
    check(ok && strstr(r, "process 1") && strstr(r, "process 2"), "list_processes");
    free(r);
    r = call(t, "stop_process", "{\"id\": 1}", &ok);
    check(ok && strstr(r, "stopped") && strstr(r, "exited"), "stop_process");
    free(r);
    r = call(t, "read_process", "{\"id\": 9}", &ok);
    check(!ok && strstr(r, "no background process"), "unknown id");
    free(r);

    r = call(t, "write_file", "{\"path\": \"e.txt\", \"content\": \"alpha\\nbeta\\nbeta\\n\"}", &ok);
    free(r);
    r = call(t, "edit_file", "{\"path\": \"e.txt\", \"old_text\": \"beta\", \"new_text\": \"gamma\"}", &ok);
    check(!ok && strstr(r, "2 times"), "edit_file refuses an ambiguous edit");
    free(r);
    r = call(t, "edit_file", "{\"path\": \"e.txt\", \"old_text\": \"beta\", \"new_text\": \"gamma\", \"replace_all\": true}", &ok);
    check(ok && strstr(r, "2 replacements"), "edit_file replace_all");
    free(r);
    r = call(t, "edit_file", "{\"path\": \"e.txt\", \"old_text\": \"delta\", \"new_text\": \"x\"}", &ok);
    check(!ok && strstr(r, "not found"), "edit_file reports missing text");
    free(r);

    Json *a = json_parse("{\"command\": \"git push origin main\"}", 35, NULL, 0);
    check(tools_danger("run_command", a) != NULL, "git push needs approval");
    json_free(a);
    const char *rm = "{\"command\": \"Remove-Item build -Recurse -Force\"}";
    a = json_parse(rm, (int)strlen(rm), NULL, 0);
    check(tools_danger("run_command", a) != NULL, "Remove-Item -Recurse needs approval");
    json_free(a);
    a = json_parse("{\"command\": \"git status\"}", 25, NULL, 0);
    check(tools_danger("run_command", a) == NULL, "git status does not");
    json_free(a);

    tools_free(t);
    printf("\n%s (%d failed)\n", fails ? "FAILED" : "all passed", fails);
    return fails != 0;
}
