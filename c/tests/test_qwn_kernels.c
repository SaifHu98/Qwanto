#include "../qwanto_kernels.h"
#include "../qwanto_native.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    QwnModel model;
    const char *error = NULL;
    if (qwn_open(argv[1], &model, &error) != 0) return 3;
    const QwnTensorDesc *tensor = qwn_find(&model, "weight");
    if (!tensor) { qwn_close(&model); return 4; }
    float row[256];
    if (qwn_row_f32(&model, tensor, 0, row, 256) != 0) {
        qwn_close(&model); return 5;
    }
    for (int i = 0; i < 256; i++) printf("%.9g%c", row[i], i == 255 ? '\n' : ' ');
    qwn_close(&model);
    return 0;
}
