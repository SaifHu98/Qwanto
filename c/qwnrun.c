#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qwanto_decode.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static void emit(const char *s,int n,void *opaque){(void)opaque;fwrite(s,1,(size_t)n,stdout);fflush(stdout);}

typedef struct { const char *id; } ServeOut;
static void emit_mux(const char *s,int n,void *opaque){
    ServeOut *o=(ServeOut*)opaque;
    printf("DATA %s %d\n",o->id,n);fwrite(s,1,(size_t)n,stdout);putchar('\n');fflush(stdout);
}

static int serve_mode(const char *model){
    int ctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int default_max=getenv("NGEN")?atoi(getenv("NGEN")):512;
    QwnDecoder d;const char *error=NULL;
    if(qwn_decoder_open(&d,model,ctx,&error)!=0){fprintf(stderr,"qwnrun: %s\n",error?error:"open");return 1;}
    printf("\x01\x01READY\x01\x01\nSTAT 0 0 0 0\n");fflush(stdout);
    char line[512];
    while(fgets(line,sizeof(line),stdin)){
        char id[64];int slot=0,bytes=0,max_tokens=default_max;float temp=0,top_p=1;
        if(sscanf(line,"SUBMIT %63s %d %d %d %f %f",id,&slot,&bytes,&max_tokens,&temp,&top_p)==6){
            (void)slot;(void)temp;(void)top_p;
            if(bytes<0||bytes>(16<<20)){printf("ERROR %s invalid-prompt-size\n",id);fflush(stdout);continue;}
            char *prompt=(char*)malloc((size_t)bytes+1);if(!prompt){printf("ERROR %s out-of-memory\n",id);fflush(stdout);continue;}
            if(fread(prompt,1,(size_t)bytes,stdin)!=(size_t)bytes){free(prompt);break;}
            prompt[bytes]=0;if(fgetc(stdin)!='\n'){free(prompt);break;}
            int *ids=(int*)malloc((size_t)ctx*sizeof(int));if(!ids){free(prompt);printf("ERROR %s out-of-memory\n",id);fflush(stdout);continue;}
            int count=tok_encode(&d.tokenizer,prompt,bytes,ids,ctx-1);free(prompt);
            if(d.cfg.bos_id>=0&&count<ctx){memmove(ids+1,ids,(size_t)count*sizeof(int));ids[0]=d.cfg.bos_id;count++;}
            qwn_decoder_reset(&d);ServeOut out={id};clock_t start=clock();
            int generated=qwn_decoder_generate(&d,ids,count,max_tokens,temp,top_p,emit_mux,&out);free(ids);
            if(generated<0){printf("ERROR %s generation-failed\n",id);fflush(stdout);continue;}
            double sec=(double)(clock()-start)/CLOCKS_PER_SEC;
            double tps=sec>0?generated/sec:0;
            printf("DONE %s STAT %d %.3f 0 0 %d %d\n",id,generated,tps,count,generated>=max_tokens);fflush(stdout);
        }else if(sscanf(line,"CANCEL %63s",id)==1){
            printf("ERROR %s CANCELLED\n",id);fflush(stdout);
        }
    }
    qwn_decoder_close(&d);return 0;
}

int main(int argc,char **argv){
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);
    _setmode(_fileno(stdout),_O_BINARY);
#endif
    if(getenv("SERVE")){
        const char *model=getenv("SNAP");if(!model||!*model){fprintf(stderr,"SNAP missing\n");return 2;}
        return serve_mode(model);
    }
    if(argc<3){fprintf(stderr,"usage: qwnrun model.qwn 'prompt' [max_tokens] [ctx]\n");return 2;}
    int max_tokens=argc>3?atoi(argv[3]):256,ctx=argc>4?atoi(argv[4]):4096;
    QwnDecoder decoder;const char *error=NULL;
    if(qwn_decoder_open(&decoder,argv[1],ctx,&error)!=0){
        fprintf(stderr,"qwnrun: %s\n",error?error:"open failed");return 1;
    }
    int max_prompt=ctx>8?ctx-8:ctx;
    int *ids=(int*)malloc((size_t)max_prompt*sizeof(int));if(!ids)return 1;
    int count=tok_encode(&decoder.tokenizer,argv[2],(int)strlen(argv[2]),ids,max_prompt);
    if(count<=0){fprintf(stderr,"qwnrun: prompt encoded to zero tokens\n");return 1;}
    if(decoder.cfg.bos_id>=0&&count<max_prompt){memmove(ids+1,ids,(size_t)count*sizeof(int));ids[0]=decoder.cfg.bos_id;count++;}
    int rc=qwn_decoder_generate(&decoder,ids,count,max_tokens,0.0f,1.0f,emit,NULL);
    putchar('\n');free(ids);qwn_decoder_close(&decoder);return rc<0?1:0;
}
