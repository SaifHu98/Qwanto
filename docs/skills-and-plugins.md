# Skills and plugins

Qwanto Code keeps workflow skills and capability plugins local by default.
Neither is part of the five primary destinations; the desktop Settings > Agent
section contains the Skills & Plugins review surface.

## Skills

A readable skill package contains:

```text
skill.json
SKILL.md
templates/   (optional)
tests/       (optional)
```

The built-in skills are Code Review, Test and Fix, Git Commit / Branch / Pull
Request, Release Readiness, Project Memory, Documentation Writer, Local
Benchmark, Optional Web Research, and GitHub Issue Reporter. A chat prompt can
invoke a built-in skill with `@skill-name`. The timeline shows the active skill
and requested capabilities.

## Plugins

A plugin package must provide a `plugin.json` with a publisher identity,
version, SHA-256 package digest, requested capabilities, package-relative
entrypoint, and a non-empty signature. The native host validates the package
bytes and allowlisted capabilities before storing it under application data.
Installation is disabled by default; quarantine, disable, uninstall, and
diagnostic actions are explicit.

The supported capability names are:

```text
workspace.read       workspace.write      terminal.execute
git.read              git.write             github.read
github.write          network.search       model.control
diagnostics.read      secrets.access
```

Writes, command execution, remote GitHub actions, network search, model
control, and secret access require approval. Workspace paths remain contained
by the desktop permission policy, approval tokens are one-shot and bound to
the exact action, and logs use the existing secret redaction boundary.

Remote installation and updates are opt-in and must show the publisher,
permissions, checksum, source URL, and license before a package is accepted.
The current release does not execute third-party plugin code: the native
sandbox/supervisor is deliberately fail-closed until it is available. This is
preferable to claiming unrestricted plugin execution or treating a manifest's
signature field as proof of trust without a configured publisher trust store.

GitHub remains an optional external capability. Local inference never falls
back to GitHub or a cloud model, and tokens are not accepted by the browser UI,
exported diagnostics, or logs. Repository writes, issue/PR/release creation,
workflow dispatch, and pushes remain approval-gated when a future OS-keychain
GitHub backend is enabled.
