#!/usr/bin/env bash
# Linux setup and build. Engine sources and generated headers are not changed.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash build.sh [options]

Check Linux build prerequisites and compile Pithos in build/.
If prerequisites are missing, show the package commands and ask before installing.

  --yes          Allow installation of missing prerequisites without confirmation
  --no-install   Never install packages (also suitable for offline builds)
  --test         Also build the C tests and run test_shell in a temporary folder
  --download-model  Download/verify the reference 24B GGUF (13.5 GiB) in models/
  --native       Optimize for this CPU; do not use for distributable binaries
  --jobs N       Parallel build jobs, 1-256 (default: detected CPU count, max 256)
  -h, --help     Show this help

Supported package managers: apt-get, dnf, pacman, zypper.
Use CC=gcc or CC=clang to select a compiler executable. CFLAGS are passed to make.
No model, Python, OpenCL SDK or GPU driver is needed to build.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
option_error() { printf 'Error: %s\nRun bash build.sh --help for usage.\n' "$*" >&2; exit 2; }
say() { printf '\n%s\n' "$*"; }

allow_install=0
no_install=0
run_tests=0
download_model=0
native=0
jobs=
# Pin the source revision as well as the expected bytes: updates must be explicit.
model_name=LFM2.5-24B-A2B-Q4_K_M.gguf
model_size=14526258464
model_sha256=d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d
model_url=https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/0912d33141f77e10705345a6732bfd83f0252fdc/LFM2.5-24B-A2B-Q4_K_M.gguf
while (($#)); do
    case "$1" in
        --yes) allow_install=1 ;;
        --no-install) no_install=1 ;;
        --test) run_tests=1 ;;
        --download-model) download_model=1 ;;
        --native) native=1 ;;
        --jobs)
            (($# >= 2)) || option_error '--jobs needs a number.'
            jobs=$2
            if [[ ! $jobs =~ ^[1-9][0-9]{0,2}$ ]] || ((jobs > 256)); then
                option_error '--jobs must be between 1 and 256.'
            fi
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) option_error "Unknown option: $1" ;;
    esac
    shift
done
((allow_install == 0 || no_install == 0)) || option_error '--yes and --no-install cannot be combined.'
[[ $(uname -s) == Linux ]] || die 'This script is for Linux/WSL. See README.md for other systems.'
case $(uname -m) in
    x86_64|aarch64|arm64) ;;
    *) die 'This script supports x86-64 and ARM64. For other architectures, consult the Makefile.' ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ -f "$repo/Makefile" && -f "$repo/src/main.c" ]] || die 'Keep build.sh in the Pithos source folder.'
if [[ -z $jobs ]]; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')
    [[ $jobs =~ ^[1-9][0-9]{0,5}$ ]] || jobs=2
    ((jobs <= 256)) || jobs=256
fi

probe_dir=
test_dir=
model_lock=
cleanup() {
    [[ -z $model_lock ]] || rmdir -- "$model_lock" 2>/dev/null || true
    [[ -z $probe_dir ]] || rm -rf -- "$probe_dir"
    [[ -z $test_dir ]] || rm -rf -- "$test_dir"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/pithos-build.XXXXXXXX") || die 'Cannot create a temporary build-check folder.'
cat > "$probe_dir/probe.c" <<'EOF'
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <dlfcn.h>
_Static_assert(sizeof(uint64_t) == 8, "C11 integer types required");
static void *worker(void *p) { return p; }
int main(int argc, char **argv) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, NULL)) return 1;
    if (pthread_join(thread, NULL)) return 1;
    void *handle = dlopen(NULL, RTLD_LAZY);
    if (!handle) return 1;
    dlclose(handle);
    printf("%f\n", cos((double)argc));
    return 0;
}
EOF

compiler=
make_tool=
find_tools() {
    compiler=
    make_tool=
    if [[ -n ${CC:-} ]]; then
        command -v "$CC" >/dev/null 2>&1 || die "Compiler CC=$CC not found. Install it or unset CC. CC must name one executable."
        compiler=$CC
    else
        for candidate in cc gcc clang; do
            if command -v "$candidate" >/dev/null 2>&1; then compiler=$candidate; break; fi
        done
    fi
    for candidate in make gmake; do
        if command -v "$candidate" >/dev/null 2>&1 && [[ $("$candidate" --version 2>/dev/null) == *'GNU Make'* ]]; then
            make_tool=$candidate
            break
        fi
    done
}

probe_compiler() {
    [[ -n $compiler ]] || return 1
    "$compiler" -std=c11 -D_GNU_SOURCE "$probe_dir/probe.c" -o "$probe_dir/probe" -lm -lpthread -ldl \
        > "$probe_dir/compiler.log" 2>&1 && "$probe_dir/probe" >> "$probe_dir/compiler.log" 2>&1
}

check_prerequisites() {
    missing=()
    need_compiler=0
    need_make=0
    need_curl=0
    find_tools
    if [[ -z $make_tool ]]; then need_make=1; missing+=('GNU Make'); fi
    if ! probe_compiler; then
        need_compiler=1
        missing+=('a working C11 compiler, C headers and system libraries')
    fi
    if ! command -v curl >/dev/null 2>&1; then
        need_curl=1
        missing+=('curl (model downloads and web tools)')
    fi
    ((${#missing[@]} == 0))
}

install_prerequisites() {
    local manager answer
    local -a packages=() privilege=()
    if command -v apt-get >/dev/null 2>&1; then
        manager=apt-get
        if ((need_compiler)); then packages+=(build-essential)
        elif ((need_make)); then packages+=(make); fi
    elif command -v dnf >/dev/null 2>&1; then
        manager=dnf
        ((need_compiler == 0)) || packages+=(gcc glibc-devel)
        ((need_make == 0)) || packages+=(make)
    elif command -v pacman >/dev/null 2>&1; then
        manager=pacman
        ((need_compiler == 0)) || packages+=(gcc glibc)
        ((need_make == 0)) || packages+=(make)
    elif command -v zypper >/dev/null 2>&1; then
        manager=zypper
        ((need_compiler == 0)) || packages+=(gcc glibc-devel)
        ((need_make == 0)) || packages+=(make)
    else
        die 'No supported package manager found. Install a C11 compiler, GNU Make, C development headers, curl and CA certificates, then run this script again.'
    fi
    # Keep an existing curl/curl-minimal installation (notably on Fedora).
    ((need_curl == 0)) || packages+=(curl)
    packages+=(ca-certificates)

    local -a install_command
    case $manager in
        apt-get) install_command=(apt-get install -y "${packages[@]}") ;;
        dnf) install_command=(dnf install -y "${packages[@]}") ;;
        pacman) install_command=(pacman -S --needed --noconfirm "${packages[@]}") ;;
        zypper) install_command=(zypper --non-interactive install --no-recommends "${packages[@]}") ;;
    esac
    say 'Required package commands (administrator privileges are used only here):'
    [[ $manager != apt-get ]] || printf '  apt-get update\n'
    printf '  '; printf '%q ' "${install_command[@]}"; printf '\n'
    ((no_install == 0)) || die 'Prerequisites are missing and --no-install was requested.'
    if ((allow_install == 0)); then
        [[ -t 0 ]] || die 'Installation needs confirmation. Run interactively, install the packages above yourself, or pass --yes.'
        read -r -p 'Install these prerequisites? [y/N] ' answer || answer=n
        case $answer in y|Y|yes|YES) ;; *) die 'Installation declined. No packages were changed.' ;; esac
    fi
    if ((EUID != 0)); then
        command -v sudo >/dev/null 2>&1 || die 'sudo is unavailable. Ask an administrator to install the packages above, then rerun as your normal user.'
        if [[ -t 0 ]]; then
            sudo -v || die 'Could not obtain administrator privileges.'
        else
            sudo -n true || die 'sudo cannot authenticate without a terminal. Install prerequisites interactively first.'
        fi
        privilege=(sudo -n)
    fi
    if [[ $manager == apt-get ]]; then
        "${privilege[@]}" apt-get update || die 'Package index refresh failed. Check your network and package sources, then retry.'
    fi
    "${privilege[@]}" "${install_command[@]}" || die 'Package installation failed. Resolve the package-manager error above, then retry. No system upgrade was requested.'
    hash -r
}

verify_model_file() {
    local size digest
    size=$(stat -Lc '%s' -- "$1") || return 1
    [[ $size == "$model_size" ]] || return 1
    digest=$(sha256sum < "$1") || return 1
    [[ ${digest%% *} == "$model_sha256" ]]
}

download_reference_model() {
    local tool directory model partial lock partial_size=0 free_kb='' available df_output required
    for tool in sha256sum stat df mkdir rmdir mv; do
        command -v "$tool" >/dev/null 2>&1 || die "--download-model requires $tool (GNU coreutils). Install it and retry."
    done
    directory=$repo/models
    mkdir -p -- "$directory" || die "Cannot create $directory. Check folder permissions."
    model=$directory/$model_name
    partial=$model.part
    lock=$model.download.lock
    if ! mkdir -- "$lock" 2>/dev/null; then
        die "Cannot acquire download lock: $lock. Another download may be running. After a crash, remove this empty lock directory only if no download is active."
    fi
    model_lock=$lock

    if [[ -e $model || -L $model ]]; then
        [[ -f $model ]] || die "Model path is not a regular file: $model"
        say 'Checking the existing reference model (size and SHA-256)...'
        verify_model_file "$model" || die "Existing model failed verification: $model. It was not changed. Move it aside or remove it before retrying."
        say "Verified model already available: $model"
        return
    fi
    [[ ! -L $partial ]] || die "Refusing a symbolic link as a partial download: $partial"
    if [[ -e $partial ]]; then
        [[ -f $partial ]] || die "Partial download is not a regular file: $partial"
        partial_size=$(stat -c '%s' -- "$partial") || die 'Cannot read the partial download size.'
    fi
    ((partial_size <= model_size)) || die "Partial download is oversized: $partial. Move it aside or remove it before retrying."

    if ((partial_size < model_size)); then
        df_output=$(LC_ALL=C df -Pk -- "$directory") || die 'Cannot check available disk space.'
        while read -r _filesystem _blocks _used available _capacity _mount; do
            [[ $available =~ ^[0-9]+$ ]] && free_kb=$available
        done <<< "$df_output"
        [[ -n $free_kb ]] || die 'Cannot determine available disk space.'
        # Existing partial data is already allocated; allow an extra 64 MiB margin.
        required=$((model_size - partial_size + 64 * 1024 * 1024))
        ((free_kb * 1024 >= required)) || die "Not enough free space: need at least $(((required + 1048575) / 1048576)) MiB more on the models filesystem."
        say "Downloading $model_name (13.5 GiB; resuming from $partial_size bytes)..."
        printf 'Source: %s\n' "$model_url"
        if ! curl --fail --location --show-error --progress-bar \
            --proto '=https' --proto-redir '=https' \
            --continue-at - --retry 5 --retry-delay 3 --connect-timeout 20 \
            --speed-limit 1024 --speed-time 60 --output "$partial" "$model_url"; then
            die "Download failed. Partial data is kept in $partial. Rerun with --download-model to resume."
        fi
    fi
    say 'Verifying downloaded model size and SHA-256...'
    verify_model_file "$partial" || die "Model verification failed. $partial was kept for inspection; move it aside or remove it and retry. No final model was created."
    # Same-filesystem rename only after verification; never replace another file.
    mv -nT -- "$partial" "$model" || die 'Could not place the verified model in models/.'
    [[ ! -e $partial ]] || die 'A model file appeared during the download. Both files were preserved; inspect models/ before retrying.'
    say "Verified model ready: $model"
}

say 'Checking Linux build prerequisites...'
if ! check_prerequisites; then
    printf 'Missing: %s\n' "${missing[@]}"
    install_prerequisites
    if ! check_prerequisites; then
        [[ ! -f "$probe_dir/compiler.log" ]] || cat "$probe_dir/compiler.log" >&2
        die 'Prerequisites are still unavailable after installation. Check the compiler errors above.'
    fi
fi

say "Building Pithos with $compiler ($jobs jobs)..."
cd -- "$repo"
# Rebuild even when a previous binary/object used another compiler or --native.
# Explicit variables also prevent Windows OS or NATIVE environment values leaking in.
make_args=(-B -j "$jobs" OS=Linux BUILD=build "CC=$compiler" NATIVE=)
((native == 0)) || make_args+=(NATIVE=1)
target=all
((run_tests == 0)) || target='test'
"$make_tool" "${make_args[@]}" "$target" || die 'Build failed. Read the compiler output above; no engine sources were changed.'
"$repo/build/pithos" --help >/dev/null || die 'The binary was built but could not run. Check its shared-library requirements.'

if ((run_tests)); then
    say 'Running shell tests in a separate temporary workspace...'
    test_dir=$(mktemp -d "${TMPDIR:-/tmp}/pithos-shell.XXXXXXXX")
    (cd -- "$test_dir" && "$repo/build/test_shell") || die 'Shell tests failed.'
    say 'Shell tests passed. Numerical/GPU tests need the reference GGUF model and were not run.'
fi

say "Build complete: $repo/build/pithos"
if ((download_model)); then
    download_reference_model
else
    printf 'No model was downloaded. Follow README.md or rerun with --download-model.\n'
fi
printf 'Then, from the Pithos folder: ./build/pithos --no-gpu -p "Hello"\n'
