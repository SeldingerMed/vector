# B1 — Isolation threat model (Phase B)

Status: frozen 2026-09-08. Code changes in this phase implement what is
marked enforced; everything else is an explicit gap with an owner.

## Tiers

- **T0 — trusted development** (default): package author runs their own
  agent/verifier locally. Local subprocess separation applies: protocol-only
  label routing, per-package cwd, timeouts, scrubbed process environment.
  T0 protects against accidents and honest-agent label leakage, not malice.
- **T1 — untrusted submission**: containerized plugin execution exists
  (`RuntimeDescriptor` kind `container` → digest-pinned image, `--network
  none`, read-only package mount, non-root, dropped capabilities,
  memory/CPU/pids caps from the digest-covered identity). CI builds the
  sandbox image and runs the live backend test digest-pinned
  (`.github/workflows/plugin-image.yml`); registry publication for hosted
  fleets is still pending. Hosted attestation (B5) still pending for
  cross-lab trust; the OSS verification contract is
  `or_audit.eval.attestation`.

1. **Oracle routing**: labels travel only to the verifier context, never in
   agent payloads (`runner.py`, `plugins.py`). A protocol-conformant agent
   cannot receive labels through the harness API.
2. **Per-package working directory**: every plugin child starts in its own
   package root (`load_*_runtime(..., cwd=root.resolve())`). This scopes
   relative file access by convention, not by enforcement (see gaps).
3. **Process environment scrubbing** (`plugins._scrubbed_plugin_env`): plugin
   children receive an enumerated allowlist, never ambient secrets. This
   stops accidental inheritance only: same-UID code can still read parent
   state through OS channels (e.g. `/proc/$PPID/environ` on Linux), so it
   is not a credential boundary against hostile code. HOME points at a
## Explicitly not enforced on the local path (gaps, not bugs)

These hold for `local` subprocess execution (T0). The `container` backend
resolves the filesystem, network, and resource rows where it runs; the
forgery and task-author rows hold everywhere until B5/governance land.

- **Filesystem**: a local agent shares a UID with the verifier and can read
  sibling task files including labels. Mitigation today: T0 only, or the
  container backend (separate mount namespace, read-only package).
- **Network**: local subprocesses have no egress policy. The container
  backend runs `--network none` and never pulls at run time. Do not
  evaluate untrusted code that must not reach the network outside it.
- **Resources**: no CPU/RAM/GPU limits on local plugin children. The
  container backend applies memory/CPU/pids caps.
- **Result forgery**: job heads are unkeyed digests; local files are
  re-stampable by anyone holding them. Holds on every backend until B5
  hosted attestation lands.
- **Task-author trust**: sandboxing an agent never validates a dishonest task

## Acceptance for B (reminder)

B1 this document. B2 one enforceable container/sandbox backend with separate
agent/verifier environments and manifest-declared transfer. B4 resource and
egress enforcement. B5 replay/publication under the same trust model. Until
then, any claim of adversarial isolation is refused.
