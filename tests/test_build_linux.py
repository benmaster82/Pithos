"""Linux build-helper regressions. Package managers are mocked; no packages change.

Run: python3 tests/test_build_linux.py
"""
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

IS_LINUX = sys.platform.startswith('linux')
if IS_LINUX:
    import pty


DISPATCHER = r'''#!/usr/bin/python3
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ['MOCK_ROOT'])
bin_dir = root / 'bin'
state = root / 'installed'
with (root / 'events.jsonl').open('a') as log:
    log.write(json.dumps(dict(name=name, args=args, cwd=os.getcwd(), uid=os.geteuid())) + '\n')
if name == 'uname':
    print(os.environ.get('MOCK_SYSTEM', 'Linux') if args == ['-s'] else os.environ.get('MOCK_ARCH', 'x86_64'))
elif name == 'getconf':
    print('4')
elif name == 'df':
    print('Filesystem 1024-blocks Used Available Capacity Mounted on')
    print('fixture 100000000 0 ' + os.environ.get('MOCK_FREE_KB', '100000000') + ' 0% /fixture')
elif name == 'curl':
    data = pathlib.Path(os.environ['MOCK_DOWNLOAD_SOURCE']).read_bytes()
    output = pathlib.Path(args[args.index('--output') + 1])
    start = output.stat().st_size if output.exists() else 0
    if os.environ.get('MOCK_DOWNLOAD_BAD'):
        data = b'x' * len(data)
    end = len(data) // 2 if os.environ.get('MOCK_DOWNLOAD_FAIL') else len(data)
    with output.open('ab') as stream:
        stream.write(data[start:end])
    sys.exit(18 if os.environ.get('MOCK_DOWNLOAD_FAIL') else 0)
elif name == 'cc':
    if os.environ.get('MOCK_PROBE_ALWAYS_FAIL') or (os.environ.get('MOCK_PROBE_FAIL') and not state.exists()):
        print('missing C development header', file=sys.stderr)
        sys.exit(1)
    output = pathlib.Path(args[args.index('-o') + 1])
    output.write_text('#!/bin/sh\nexit 0\n')
    output.chmod(0o755)
elif name == 'make':
    if args == ['--version']:
        print('GNU Make 4.4')
    else:
        if os.environ.get('MOCK_MAKE_FAIL'):
            sys.exit(2)
        build = pathlib.Path('build')
        build.mkdir(exist_ok=True)
        for tool in ('pithos', 'test_shell'):
            target = build / tool
            if not target.exists():
                target.symlink_to(bin_dir / 'dispatcher')
elif name == 'pithos':
    sys.exit(int(os.environ.get('MOCK_BINARY_FAIL', '0')))
elif name == 'test_shell':
    pathlib.Path('e.txt').write_text('scratch file')
    sys.exit(int(os.environ.get('MOCK_TEST_FAIL', '0')))
elif name == 'sudo':
    if os.environ.get('MOCK_SUDO_FAIL'):
        sys.exit(1)
    if args in (['-v'], ['-n', 'true']):
        sys.exit(0)
    sys.exit(subprocess.run(args[1:] if args[0] == '-n' else args).returncode)
elif name in ('apt-get', 'dnf', 'pacman', 'zypper'):
    if os.environ.get('MOCK_INSTALL_FAIL'):
        sys.exit(1)
    if args != ['update']:
        state.touch()
        for tool in ('cc', 'make', 'curl'):
            target = bin_dir / tool
            if not target.exists():
                target.symlink_to(bin_dir / 'dispatcher')
'''


@unittest.skipUnless(IS_LINUX, 'Linux build helper')
class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pithos-build-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'checkout with spaces'
        self.bin = self.root / 'bin'
        for path in (self.root, self.repo, self.bin, self.root / 'tmp'):
            path.mkdir(exist_ok=True)
            path.chmod(0o777)
        (self.repo / 'src').mkdir()
        (self.repo / 'src/main.c').touch()
        (self.repo / 'Makefile').touch()
        shutil.copy(Path(__file__).resolve().parents[1] / 'build.sh', self.repo / 'build.sh')
        dispatcher = self.bin / 'dispatcher'
        dispatcher.write_text(DISPATCHER)
        dispatcher.chmod(0o755)
        for tool in ('uname', 'getconf', 'cc', 'make', 'curl', 'sudo', 'apt-get', 'df'):
            (self.bin / tool).symlink_to(dispatcher)
        for tool in ('cat', 'dirname', 'mktemp', 'rm', 'mkdir', 'rmdir', 'mv', 'stat', 'sha256sum'):
            (self.bin / tool).symlink_to(shutil.which(tool))
        self.env = dict(os.environ, PATH=str(self.bin), MOCK_ROOT=str(self.root), TMPDIR=str(self.root / 'tmp'))
        for name in ('CC', 'CFLAGS', 'MAKEFLAGS', 'MFLAGS', 'NATIVE', 'OS', 'BASH_ENV', 'ENV'):
            self.env.pop(name, None)

    def run_build(self, *args, extra=None, unprivileged=False):
        env = dict(self.env, **(extra or {}))
        demote = None
        if unprivileged and os.geteuid() == 0:
            def demote():
                os.setgid(65534)
                os.setuid(65534)
        return subprocess.run(['/bin/bash', str(self.repo / 'build.sh'), *args], env=env,
                              cwd=self.root, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=20, preexec_fn=demote)

    def events(self, name):
        path = self.root / 'events.jsonl'
        if not path.exists():
            return []
        return [event for line in path.read_text().splitlines()
                if (event := json.loads(line))['name'] == name]

    def builds(self):
        return [e for e in self.events('make') if e['args'] != ['--version']]

    def configure_download(self):
        """Use a small fixture in this disposable script, never in production."""
        self.payload = b'GGUF-download-regression-fixture\n' * 32
        payload_path = self.root / 'model-payload'
        payload_path.write_bytes(self.payload)
        self.env['MOCK_DOWNLOAD_SOURCE'] = str(payload_path)
        script = self.repo / 'build.sh'
        text = script.read_text().replace('model_size=14526258464', f'model_size={len(self.payload)}')
        text = text.replace('model_sha256=d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d',
                            'model_sha256=' + hashlib.sha256(self.payload).hexdigest())
        script.write_text(text)
        directory = self.repo / 'models'
        directory.mkdir()
        self.model = directory / 'LFM2.5-24B-A2B-Q4_K_M.gguf'
        self.partial = Path(str(self.model) + '.part')
        self.lock = Path(str(self.model) + '.download.lock')

    def test_download_is_verified_before_promotion(self):
        self.configure_download()
        result = self.run_build('--no-install', '--download-model')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.lock.exists())
        args = self.events('curl')[0]['args']
        self.assertEqual(args[args.index('--output') + 1], str(self.partial))
        self.assertIn('--continue-at', args)
        self.assertIn('--fail', args)
        self.assertIn('/0912d33141f77e10705345a6732bfd83f0252fdc/', args[-1])

    def test_existing_valid_model_is_verified_and_reused(self):
        self.configure_download()
        self.model.write_bytes(self.payload)
        result = self.run_build('--no-install', '--download-model')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('already available', result.stdout)
        self.assertFalse(self.events('curl'))
        self.assertFalse(self.lock.exists())

    def test_existing_invalid_model_is_preserved(self):
        self.configure_download()
        data = b'x' * len(self.payload)
        self.model.write_bytes(data)
        result = self.run_build('--no-install', '--download-model')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.model.read_bytes(), data)
        self.assertFalse(self.events('curl'))
        self.assertFalse(self.lock.exists())

    def test_failed_transfer_keeps_partial_and_next_run_resumes(self):
        self.configure_download()
        result = self.run_build('--no-install', '--download-model', extra={'MOCK_DOWNLOAD_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.partial.read_bytes(), self.payload[:len(self.payload) // 2])
        self.assertFalse(self.model.exists())
        self.assertFalse(self.lock.exists())
        result = self.run_build('--no-install', '--download-model')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())

    def test_checksum_failure_never_creates_final_model(self):
        self.configure_download()
        result = self.run_build('--no-install', '--download-model', extra={'MOCK_DOWNLOAD_BAD': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('verification failed', result.stderr)
        self.assertFalse(self.model.exists())
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.lock.exists())

    def test_completed_partial_is_verified_without_network(self):
        self.configure_download()
        self.partial.write_bytes(self.payload)
        result = self.run_build('--no-install', '--download-model', extra={'MOCK_FREE_KB': '0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertFalse(self.events('curl'))

    def test_oversized_partial_is_preserved(self):
        self.configure_download()
        self.partial.write_bytes(self.payload + b'extra')
        result = self.run_build('--no-install', '--download-model')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('oversized', result.stderr)
        self.assertFalse(self.events('curl'))
        self.assertTrue(self.partial.exists())

    def test_insufficient_space_stops_before_transfer(self):
        self.configure_download()
        result = self.run_build('--no-install', '--download-model', extra={'MOCK_FREE_KB': '0'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Not enough free space', result.stderr)
        self.assertFalse(self.events('curl'))
        self.assertFalse(self.model.exists())
        self.assertFalse(self.lock.exists())

    def test_download_lock_is_respected(self):
        self.configure_download()
        self.lock.mkdir()
        result = self.run_build('--no-install', '--download-model')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.lock.is_dir())
        self.assertFalse(self.events('curl'))

    def test_partial_symlink_is_never_written(self):
        self.configure_download()
        target = self.root / 'untouched'
        target.write_bytes(b'keep me')
        self.partial.symlink_to(target)
        result = self.run_build('--no-install', '--download-model')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), b'keep me')
        self.assertFalse(self.events('curl'))

    def test_real_curl_resumes_and_refuses_a_server_ignoring_ranges(self):
        self.configure_download()
        (self.bin / 'curl').unlink()
        (self.bin / 'curl').symlink_to(shutil.which('curl'))
        self.env.update(NO_PROXY='127.0.0.1', no_proxy='127.0.0.1')
        payload = self.payload
        for supports_range in (True, False):
            with self.subTest(supports_range=supports_range):
                self.model.unlink(missing_ok=True)
                self.partial.write_bytes(payload[:100])
                requests = []

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        header = self.headers.get('Range', '')
                        requests.append(header)
                        offset = int(header.removeprefix('bytes=').removesuffix('-'))
                        if supports_range:
                            self.send_response(206)
                            self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
                        else:
                            offset = 0
                            self.send_response(200)
                        self.send_header('Content-Length', str(len(payload) - offset))
                        self.end_headers()
                        self.wfile.write(payload[offset:])

                    def log_message(self, *args):
                        pass

                with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    # This disposable test copy uses a local HTTP fixture. Production stays HTTPS-only.
                    script = self.repo / 'build.sh'
                    lines = script.read_text().splitlines()
                    text = '\n'.join(f'model_url=http://127.0.0.1:{server.server_port}/model' if line.startswith('model_url=')
                                     else line for line in lines) + '\n'
                    script.write_text(text.replace("'=https'", "'=http,https'"))
                    try:
                        result = self.run_build('--no-install', '--download-model')
                    finally:
                        server.shutdown()
                        thread.join(timeout=5)
                self.assertEqual(requests, ['bytes=100-'])
                if supports_range:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.model.read_bytes(), payload)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.model.exists())
                    self.assertEqual(self.partial.read_bytes(), payload[:100])
                self.assertFalse(self.lock.exists())

    def test_ready_environment_builds_from_other_directory_with_spaces(self):
        result = self.run_build('--no-install', '--jobs', '2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.builds()[0]['cwd'], str(self.repo))
        self.assertIn('-B', self.builds()[0]['args'])
        self.assertIn('all', self.builds()[0]['args'])
        self.assertFalse(self.events('apt-get'))
        self.assertFalse(self.events('sudo'))
        self.assertEqual(list((self.root / 'tmp').iterdir()), [])

    def test_default_overrides_windows_and_native_environment(self):
        result = self.run_build('--no-install', extra={'OS': 'Windows_NT', 'NATIVE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('OS=Linux', self.builds()[0]['args'])
        self.assertIn('NATIVE=', self.builds()[0]['args'])
        self.assertNotIn('NATIVE=1', self.builds()[0]['args'])

    def test_native_is_explicit(self):
        result = self.run_build('--no-install', '--native')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.builds()[0]['args'][-2:], ['NATIVE=1', 'all'])

    def test_invalid_options_do_not_install_or_build(self):
        for args in (('--jobs', '0'), ('--jobs', '257'), ('--jobs', '-1'), ('--jobs',),
                     ('--unknown',), ('--yes', '--no-install')):
            with self.subTest(args=args):
                self.assertEqual(self.run_build(*args).returncode, 2)
        self.assertFalse(self.builds())
        self.assertFalse(self.events('apt-get'))

    def test_non_linux_and_unsupported_architecture_fail_early(self):
        for extra in ({'MOCK_SYSTEM': 'Darwin'}, {'MOCK_ARCH': 'i686'}):
            with self.subTest(extra=extra):
                self.assertNotEqual(self.run_build(extra=extra).returncode, 0)
        self.assertFalse(self.builds())

    def test_no_install_and_unattended_default_never_install(self):
        (self.bin / 'curl').unlink()
        for args in (('--no-install',), ()):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_build(*args).returncode, 0)
        self.assertFalse(self.events('apt-get'))
        self.assertFalse(self.builds())

    def test_package_manager_commands_and_recheck(self):
        for manager in ('apt-get', 'dnf', 'pacman', 'zypper'):
            with self.subTest(manager=manager):
                for candidate in ('apt-get', 'dnf', 'pacman', 'zypper', 'curl'):
                    (self.bin / candidate).unlink(missing_ok=True)
                (self.bin / manager).symlink_to(self.bin / 'dispatcher')
                result = self.run_build('--yes')
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.events(manager)
                self.assertTrue(any('curl' in e['args'] for e in commands))
                self.assertFalse(any('upgrade' in e['args'] or '-Sy' in e['args'] for e in commands))
                if manager == 'apt-get':
                    self.assertEqual(commands[0]['args'], ['update'])

    def test_interactive_install_requires_explicit_yes(self):
        for answer in (b'n\n', b'y\n'):
            with self.subTest(answer=answer):
                (self.bin / 'curl').unlink(missing_ok=True)
                master, slave = pty.openpty()
                try:
                    process = subprocess.Popen(['/bin/bash', str(self.repo / 'build.sh')],
                                               env=self.env, cwd=self.root, stdin=slave,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    os.write(master, answer)
                    stdout, stderr = process.communicate(timeout=20)
                    if answer == b'n\n':
                        self.assertNotEqual(process.returncode, 0)
                        self.assertIn(b'Installation declined', stderr)
                        self.assertFalse(self.events('apt-get'))
                        self.assertFalse(self.builds())
                    else:
                        self.assertEqual(process.returncode, 0, stderr)
                        self.assertIn(b'Build complete', stdout)
                        self.assertTrue(self.events('apt-get'))
                finally:
                    os.close(master)
                    os.close(slave)

    def test_failed_header_probe_is_repaired_before_build(self):
        result = self.run_build('--yes', extra={'MOCK_PROBE_FAIL': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.events('apt-get'))
        self.assertTrue(self.builds())
        install = self.events('apt-get')[-1]['args']
        self.assertIn('build-essential', install)
        self.assertNotIn('curl', install)

    def test_unrepaired_header_probe_prevents_build(self):
        result = self.run_build('--yes', extra={'MOCK_PROBE_ALWAYS_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('still unavailable', result.stderr)
        self.assertIn('missing C development header', result.stderr)
        self.assertFalse(self.builds())

    def test_custom_missing_compiler_does_not_install_another(self):
        result = self.run_build('--yes', extra={'CC': 'missing-compiler'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CC=missing-compiler', result.stderr)
        self.assertFalse(self.events('apt-get'))

    def test_unknown_package_manager_gives_manual_instructions(self):
        (self.bin / 'apt-get').unlink()
        (self.bin / 'curl').unlink()
        result = self.run_build('--yes')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Install a C11 compiler', result.stderr)
        self.assertFalse(self.builds())

    def test_install_failure_stops_before_build(self):
        (self.bin / 'curl').unlink()
        result = self.run_build('--yes', extra={'MOCK_INSTALL_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.builds())

    def test_sudo_is_only_used_for_package_setup(self):
        (self.bin / 'curl').unlink()
        result = self.run_build('--yes', unprivileged=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.events('sudo'))
        self.assertNotEqual(self.builds()[0]['uid'], 0)
        self.assertFalse(any('make' in e['args'] for e in self.events('sudo')))

    def test_missing_sudo_credentials_stop_before_install(self):
        (self.bin / 'curl').unlink()
        result = self.run_build('--yes', unprivileged=True, extra={'MOCK_SUDO_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events('apt-get'))

    def test_build_failure_is_not_reported_as_success(self):
        result = self.run_build('--no-install', extra={'MOCK_MAKE_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Build failed', result.stderr)
        self.assertNotIn('Build complete:', result.stdout)

    def test_binary_load_failure_is_reported(self):
        result = self.run_build('--no-install', extra={'MOCK_BINARY_FAIL': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('could not run', result.stderr)

    def test_shell_tests_use_isolated_workspace_and_propagate_failure(self):
        for failed in ('0', '1'):
            with self.subTest(failed=failed):
                result = self.run_build('--no-install', '--test', extra={'MOCK_TEST_FAIL': failed})
                self.assertEqual(result.returncode, int(failed), result.stderr)
                event = self.events('test_shell')[-1]
                self.assertNotEqual(event['cwd'], str(self.repo))
                self.assertFalse(Path(event['cwd']).exists())
                self.assertFalse((self.repo / 'e.txt').exists())
                self.assertEqual(list((self.root / 'tmp').iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
