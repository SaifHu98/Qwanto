#ifndef QWN_SPECULATIVE_H
#define QWN_SPECULATIVE_H

#include <stddef.h>
#include <stdint.h>
#include "qwanto_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    QwnDecoder *target;       /* High-capacity target model (e.g. 4B / 8B / 70B) */
    QwnDecoder *draft;        /* Ultra-fast draft model (e.g. 1.5B) */
    int         gamma;        /* Number of speculative tokens per draft pass (default 4..6) */
    int         total_drafted;
    int         total_accepted;
    float       temperature;
    float       top_p;
} QwnSpecContext;

/* Initialize speculative decoding context */
int qwn_speculative_init(QwnSpecContext *ctx, QwnDecoder *target, QwnDecoder *draft, int gamma);

/* Execute speculative generation with target parallel verification and rejection sampling */
int qwn_speculative_generate(QwnSpecContext *ctx,
                            const int *prompt, int prompt_count,
                            int max_new_tokens,
                            float temperature, float top_p,
                            void(*callback)(const char*, int, void*),
                            void *opaque);

/* Return empirical acceptance rate of drafted tokens */
float qwn_speculative_acceptance_rate(const QwnSpecContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWN_SPECULATIVE_H */
