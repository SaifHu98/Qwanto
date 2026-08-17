import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from tools.qwn_convert import DT_BYTES, DT_F32, write_qwn


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


def matrix(rows, cols, seed):
    return [(((r * cols + c + seed) % 11) - 5) * 0.025
            for r in range(rows) for c in range(cols)]


def build_fixture(path):
    d, inter, vocab = 4, 8, 6
    cfg = {"hidden_size": d, "intermediate_size": inter,
           "num_hidden_layers": 1, "num_attention_heads": 1,
           "num_key_value_heads": 1, "head_dim": d, "vocab_size": vocab,
           "max_position_embeddings": 8, "rms_norm_eps": 1e-5,
           "rope_theta": 10000.0, "bos_token_id": -1, "eos_token_id": 5,
           "tie_word_embeddings": False}
    tok = {"model": {"type": "BPE", "vocab": {chr(97+i): i for i in range(vocab)},
                     "merges": []}, "added_tokens": []}
    tensors = [
        {"name": "__qwn.config", "dtype": DT_BYTES, "shape": (len(json.dumps(cfg).encode()),), "payload": json.dumps(cfg).encode()},
        {"name": "__qwn.tokenizer", "dtype": DT_BYTES, "shape": (len(json.dumps(tok).encode()),), "payload": json.dumps(tok).encode()},
        {"name": "model.embed_tokens.weight", "dtype": DT_F32, "shape": (d, vocab), "payload": f32(matrix(vocab,d,1))},
        {"name": "model.layers.0.input_layernorm.weight", "dtype": DT_F32, "shape": (d,), "payload": f32([1.0,1.1,.9,1.05])},
        {"name": "model.layers.0.self_attn.q_proj.weight", "dtype": DT_F32, "shape": (d,d), "payload": f32(matrix(d,d,2))},
        {"name": "model.layers.0.self_attn.k_proj.weight", "dtype": DT_F32, "shape": (d,d), "payload": f32(matrix(d,d,3))},
        {"name": "model.layers.0.self_attn.v_proj.weight", "dtype": DT_F32, "shape": (d,d), "payload": f32(matrix(d,d,4))},
        {"name": "model.layers.0.self_attn.o_proj.weight", "dtype": DT_F32, "shape": (d,d), "payload": f32(matrix(d,d,5))},
        {"name": "model.layers.0.post_attention_layernorm.weight", "dtype": DT_F32, "shape": (d,), "payload": f32([.95,1.0,1.05,.9])},
        {"name": "model.layers.0.mlp.gate_proj.weight", "dtype": DT_F32, "shape": (d,inter), "payload": f32(matrix(inter,d,6))},
        {"name": "model.layers.0.mlp.up_proj.weight", "dtype": DT_F32, "shape": (d,inter), "payload": f32(matrix(inter,d,7))},
        {"name": "model.layers.0.mlp.down_proj.weight", "dtype": DT_F32, "shape": (inter,d), "payload": f32(matrix(d,inter,8))},
        {"name": "model.norm.weight", "dtype": DT_F32, "shape": (d,), "payload": f32([1.0,.95,1.1,.9])},
        {"name": "lm_head.weight", "dtype": DT_F32, "shape": (d,vocab), "payload": f32(matrix(vocab,d,9))},
    ]
    write_qwn(path,tensors,arch_dims=(d,inter,1,1,d,1,vocab,8))
    return tensors


def rms(x,w,eps=1e-5):
    inv=1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps)
    return [x[i]*inv*w[i] for i in range(len(x))]


def mm(x,w,rows,cols):
    return [sum(x[k]*w[n*cols+k] for k in range(cols)) for n in range(rows)]


def softmax(x):
    peak=max(x);vals=[math.exp(v-peak) for v in x];s=sum(vals)
    return [v/s for v in vals]


def python_reference():
    d,inter,vocab=4,8,6
    emb=matrix(vocab,d,1);qw=matrix(d,d,2);kw=matrix(d,d,3);vw=matrix(d,d,4)
    ow=matrix(d,d,5);gw=matrix(inter,d,6);uw=matrix(inter,d,7)
    dw=matrix(d,inter,8);head=matrix(vocab,d,9)
    n1=[1.0,1.1,.9,1.05];n2=[.95,1.0,1.05,.9];nf=[1.0,.95,1.1,.9]
    keys=[];values=[];logits=None
    for pos,token in enumerate((1,2)):
        x=emb[token*d:(token+1)*d]
        xb=rms(x,n1);q=mm(xb,qw,d,d);k=mm(xb,kw,d,d);v=mm(xb,vw,d,d)
        half=d//2
        for vec in (q,k):
            for i in range(half):
                angle=pos*(10000.0**(-2.0*i/d));c=math.cos(angle);s=math.sin(angle)
                a,b=vec[i],vec[i+half];vec[i]=a*c-b*s;vec[i+half]=a*s+b*c
        keys.append(k);values.append(v)
        score=softmax([sum(q[i]*keys[t][i] for i in range(d))/math.sqrt(d) for t in range(pos+1)])
        ctx=[sum(score[t]*values[t][i] for t in range(pos+1)) for i in range(d)]
        att=mm(ctx,ow,d,d);x=[x[i]+att[i] for i in range(d)]
        xb=rms(x,n2);gate=mm(xb,gw,inter,d);up=mm(xb,uw,inter,d)
        hidden=[(gate[i]/(1+math.exp(-gate[i])))*up[i] for i in range(inter)]
        down=mm(hidden,dw,d,inter);x=[x[i]+down[i] for i in range(d)]
        logits=mm(rms(x,nf),head,vocab,d)
    return logits


class NativeDecoderTest(unittest.TestCase):
    @staticmethod
    def compile(clang, exe, main):
        cmd=[clang,"-std=c11","-O2","-D_CRT_SECURE_NO_WARNINGS","-Wno-unused-function",
             str(main),str(HERE/"qwn_runtime_config.c"),str(HERE/"qwanto_decode.c"),str(HERE/"qwanto_native.c"),
             str(HERE/"qwanto_kernels.c"),str(HERE/"qwanto_turboquant.c"),str(HERE/"qwanto_thinking.c"),str(HERE/"qwn_speculative.c"),str(HERE/"qwanto_agentic.c"),str(HERE/"qwanto_autopilot.c"),str(HERE/"qwanto_gpu.c"),str(HERE/"qwanto_bitdecoding.c"),str(HERE/"qwanto_jetspec.c"),str(HERE/"qwanto_talon.c"),str(HERE/"qwanto_sliminfer.c"),str(HERE/"qwanto_pquant.c"),str(HERE/"qwanto_littlebit.c"),str(HERE/"qwn_paged_kv.c"),"-o",exe]
        if sys.platform == "linux":
            cmd.append("-fopenmp")
            cmd.append("-D_GNU_SOURCE")
        if os.name != "nt":
            cmd.extend(["-lm", "-lpthread", "-ldl"])
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                "Native decoder compilation failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def run_native(self, command):
        try:
            return subprocess.check_output(command, text=True).strip()
        except OSError as error:
            if getattr(error, "winerror", None) == 4551:
                self.skipTest("Windows Application Control blocked the temporary test executable")
            raise

    def test_two_token_logits_are_finite_and_repeatable(self):
        clang=shutil.which("clang")
        if not clang:self.skipTest("clang not installed")
        with tempfile.TemporaryDirectory() as td:
            model=os.path.join(td,"tiny.qwn");exe=os.path.join(td,"decode.exe" if os.name=="nt" else "decode")
            build_fixture(model)
            self.compile(clang,exe,HERE/"tests"/"test_qwanto_decode.c")
            out=self.run_native([exe,model])
            logits=[float(v) for v in out.split()]
            ref=python_reference()
            self.assertEqual(len(logits),len(ref))
            for a,b in zip(logits,ref):
                self.assertTrue(math.isfinite(a))
                self.assertAlmostEqual(a,b,places=2)

    def test_persistent_openai_engine_protocol(self):
        clang=shutil.which("clang")
        if not clang:self.skipTest("clang not installed")
        with tempfile.TemporaryDirectory() as td:
            model=os.path.join(td,"tiny.qwn");exe=os.path.join(td,"qwnrun.exe" if os.name=="nt" else "qwnrun")
            build_fixture(model)
            self.compile(clang, exe, HERE/"qwnrun.c")
            try:
                proc=subprocess.Popen([exe,model,"--serve"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            except OSError as error:
                if getattr(error, "winerror", None) == 4551:
                    self.skipTest("Windows Application Control blocked the temporary test executable")
                raise
            try:
                proc.stdin.write("PING\n");proc.stdin.flush()
                self.assertEqual(proc.stdout.readline().strip(),"PONG")
                proc.stdin.write("CONFIG\n");proc.stdin.flush()
                line=proc.stdout.readline().strip()
                self.assertTrue(line.startswith("CONFIG "),line)
                self.assertIn("dim=4",line)
                proc.stdin.write("FORWARD 1\n");proc.stdin.flush()
                self.assertEqual(proc.stdout.readline().strip(),"LOGITS 6")
                logits1=[float(proc.stdout.readline().strip()) for _ in range(6)]
                proc.stdin.write("FORWARD 2\n");proc.stdin.flush()
                self.assertEqual(proc.stdout.readline().strip(),"LOGITS 6")
                logits2=[float(proc.stdout.readline().strip()) for _ in range(6)]
                ref=python_reference()
                for a,b in zip(logits2,ref):
                    self.assertAlmostEqual(a,b,places=2)
            finally:
                proc.terminate();proc.wait()


if __name__=="__main__":
    unittest.main()
