#include <stdio.h>
#include <stdlib.h>
#include "../qwanto_decode.h"

int main(int argc,char **argv){
    if(argc!=2)return 2;
    QwnDecoder d;const char *error=NULL;
    if(qwn_decoder_open(&d,argv[1],8,&error)!=0){fprintf(stderr,"%s\n",error?error:"open");return 1;}
    const float *logits=NULL;
    if(qwn_decoder_forward(&d,1,&logits)||qwn_decoder_forward(&d,2,&logits))return 1;
    for(int i=0;i<d.cfg.vocab;i++)printf("%.9g%c",logits[i],i+1==d.cfg.vocab?'\n':' ');
    qwn_decoder_close(&d);return 0;
}
