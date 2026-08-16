# Troubleshooting

## The dashboard says the gateway is not running

Start the gateway in one terminal and the web UI in another. Use
`http://127.0.0.1:8000/v1` as the UI base URL. Probe again after the gateway
prints its listening address.

## The dashboard says the wrong server is selected

The configured URL answered like a static web server, commonly the Vite server
on port 5173 or another service on port 8080. The gateway health endpoint is
`http://127.0.0.1:8000/health`, while models/config/telemetry use the `/v1`
base. The UI intentionally does not treat a static 404 as a gateway.

## No model is recommended

Place a real `.qwn` file in a discovered local path, then reconnect. The file
must pass structural validation, be supported by the available `qwnrun`, and
fit the reported host resources. A GGUF or source checkpoint is not a native
runtime model until it has been converted and validated.

## Benchmark values are unavailable

Run the reproducible local harness with a real `.qwn` and `qwnrun`, then open
Benchmark evidence. Missing, zero-token, invalid, or non-measured results are
shown as unavailable rather than projected. TTFT is unavailable when the
harness does not observe a first streamed data frame.
