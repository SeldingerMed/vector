# B1 — Isolation threat model (Phase B)

Status: frozen 2026-09-08. Code changes in this phase implement what is
marked enforced; everything else is an explicit gap with an owner.

## Tiers

- **T0 — trusted development** (default): package author runs their own
  agent/verifier locally. Local subprocess separation applies: protocol-only
  label routing, per-package cwd, timeouts, scrubbed process environment.
  T0 protects against accidents and honest-agent label leakage, not malice.
- **T1 — untrusted submission** (not implemented): third-party code/weights
  scored for publication. Requires a container/sandbox backend, separate
  agent and verifier filesystems with declared artifact transfer, enforced
  egress and resource policy, and hosted attestation. No local-subprocess
  configuration may claim T1.

## What is enforced (T0)

1. **Oracle routing**: labels travel only to the verifier context, never in
   agent payloads (`runner.py`, `plugins.py`). A protocol-conformant agent
   cannot receive labels through the harness API.
2. **Process environment scrubbing** (`plugins._scrubbed_plugin_env`): plugin
   children receive an allowlist (PATH, temp dirs, locale, GPU runtime
   discovery), never ambient secrets. Endpoint credentials in the parent
   process do not reach evaluated code.
3. **Timeouts and cleanup**: bounded requests, kill on expiry, pipe cleanup.

## Explicitly not enforced locally (gaps, not bugs)

- **Filesystem**: agent and verifier share a UID; a malicious agent can read
  sibling task files including labels. Mitigation today: T0 only; T1 needs
  separate mounts (B2).
- **Network**: no egress policy on local subprocesses. Do not evaluate
  untrusted code that must not reach the network (B4).
- **Resources**: no CPU/RAM/GPU limits on plugin children (B4/H).
- **Result forgery**: job heads are unkeyed digests; local files are
  re-stampable. Cross-lab trust needs hosted attestation (B5), not hashing.
- **Task-author trust**: sandboxing an agent never validates a dishonest task
  verifier. Benchmark tasks need review/governance, not just isolation.

## Acceptance for B (reminder)

B1 this document. B2 one enforceable container/sandbox backend with separate
agent/verifier environments and manifest-declared transfer. B4 resource and
egress enforcement. B5 replay/publication under the same trust model. Until
then, any claim of adversarial isolation is refused.
