# Local gateway API

Run the Python gateway separately from the web development server:

```sh
python c/coli web --model path/to/model.qwn --host 127.0.0.1 --port 8000 --no-browser
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

Configure the UI with `http://127.0.0.1:8000/v1`. Health is deliberately the
root endpoint `/health`; the control-plane endpoints stay under `/v1`:

| Purpose | Method and path |
| --- | --- |
| Gateway health and endpoint discovery | `GET /health` |
| OpenAI-compatible models | `GET /v1/models` |
| Qwanto configuration | `GET /v1/qwanto/config` |
| Qwanto telemetry | `GET /v1/qwanto/telemetry` |

Successful control-plane responses include `schema_version: "1"`. The health
response also identifies the Qwanto gateway and its API version. The web UI
does not request model, configuration, or telemetry data until health succeeds;
a static server on port 8080 is reported as the wrong server rather than as a
working gateway.

The browser calls HTTP only. Local filesystem import, conversion, downloads,
and model activation are performed by the gateway after its own validation and
path-safety checks.
