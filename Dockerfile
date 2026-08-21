# Multi-stage production Dockerfile for Qwanto Ultra Inference Engine
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    libomp-dev \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Compile native binary with OpenMP, AVX2/AVX-512 vectorization
RUN make -C c qwnrun || \
    clang -O3 -march=x86-64-v3 -fopenmp \
    -I/usr/include -Ic \
    c/qwnrun.c c/qwn_runtime_config.c c/qwanto_decode.c c/qwanto_native.c c/qwanto_kernels.c \
    c/qwanto_turboquant.c c/qwanto_thinking.c c/qwanto_speculative.c \
    c/qwanto_turboquant.c c/qwanto_thinking.c c/qwn_speculative.c \
    c/qwanto_agentic.c c/qwanto_autopilot.c c/qwn_paged_kv.c \
    -lm -lomp \
    -o c/qwnrun

# Production runtime stage
FROM ubuntu:22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    python3 \
    python3-pip \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/v1/models || exit 1

ENTRYPOINT ["python3", "c/openai_server.py", "--host", "0.0.0.0", "--port", "8000"]
