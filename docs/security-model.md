# Qwanto security model

Qwanto’s default profile is local-first. The trust boundary is the user’s
machine and the explicitly selected local workspace.

## Gateway

- The default bind address is `127.0.0.1`.
- Non-loopback binding requires an explicit API key in normal operation.
- API keys use constant-time comparison and are not persisted by the web UI.
- JSON responses retain `nosniff`, `DENY`, and strict referrer headers.
- CORS is allowlisted by default; wildcard CORS is an explicit configuration.
- User-supplied filesystem paths pass `_is_safe_path()` before filesystem
  mutation. External runtime downloads are opt-in.

## Desktop agent

- The workspace root is canonicalized and every path/cwd is checked against it.
- Plan mode cannot mutate files or execute commands.
- Mutating tools require short-lived, single-use approval tokens whose hash
  covers the exact arguments and workspace.
- Commands use explicit argv and do not invoke a shell. Output is bounded and
  secrets are redacted before session persistence.
- The desktop resource contains only the native executable. Model files are
  user-managed data and are never included in packages.

## Browser boundary

The browser has HTTP access only. It cannot use Tauri IPC, launch `qwnrun`, or
read arbitrary local files. Operators should treat a non-loopback endpoint as a
separate trusted service and configure authentication and CORS deliberately.

## Known limitations

The default gateway is HTTP on loopback, not TLS. Remote use requires a trusted
TLS reverse proxy, a strong API key, and a narrowly scoped CORS policy. Hardware
telemetry is incomplete on hosts without supported sensors and is reported as
unavailable rather than inferred.
