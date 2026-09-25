# Experimental SM120 feedback

For install or startup trouble, run:

```bash
./escha doctor --model e3 --model-path /path/to/model.gguf --report doctor.json
```

Attach `doctor.json` to an install issue. The report stays local until you
attach it and contains hashes, GPU/driver details, startup health, a short
control pass, and observed memory use. It excludes prompts, generated text,
credentials, full paths, GPU serials, and IP addresses. Choose W2 in place of
E3 if that is the model you installed.

**Install issue fields:** E3 or W2; Linux distribution or WSL; GPU and VRAM;
install step that failed; exact error category shown by doctor; expected and
actual behavior; whether an older immutable asset still starts.

**Quality or performance issue fields:** E3 or W2; model SHA-256; profile and
context length; prompt and generated token counts; median prefill/decode over
five warm runs; draft acceptance; `cache_n` and `cache_reason` for repeated
prompts; observed peak VRAM; expected and actual
behavior. Share a prompt or output only if you choose to disclose it. Mention
the comparison runtime and its model hash when claiming a speed difference.

Rollback: stop the server with Ctrl-C and run your previous installed asset
from its own directory. This archive does not replace or remove prior assets.
