#!/usr/bin/env python3
"""Translation worker — runs INSIDE the isolated translation venv, which has
its OWN pinned (likely older) transformers/torch/IndicTransToolkit versions,
fully decoupled from the main app's environment. This script is never
imported by the main app; it's only ever invoked as a subprocess by
translation.py, and talks back over stdin/stdout using one JSON object per
line.

Protocol:
  Startup:  prints {"ready": true} on success, or {"ready": false, "error": "..."}
            and exits if the model fails to load.
  Requests: one line in  -> {"sentences": ["...", "..."]}
            one line out -> {"ok": true, "translations": ["...", "..."]}
                            or {"ok": false, "error": "..."}
"""
import json
import sys
import traceback

SRC_LANG = "eng_Latn"
TGT_LANG = "mar_Deva"


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ready": False,
                          "error": "usage: translation_worker.py <model_name> <device>"}),
             flush=True)
        return

    model_name, device = sys.argv[1], sys.argv[2]

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        dtype = torch.float32 if device == "cpu" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()
        ip = IndicProcessor(inference=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({"ready": False,
                          "error": f"{type(e).__name__}: {e}\n{tb[-3000:]}"}), flush=True)
        return

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            sentences = req.get("sentences", [])
            if not sentences:
                print(json.dumps({"ok": True, "translations": []}), flush=True)
                continue
            batch = ip.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
            inputs = tokenizer(batch, truncation=True, padding="longest",
                              return_tensors="pt", max_length=384).to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_length=384, num_beams=5,
                                           use_cache=True)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated = ip.postprocess_batch(decoded, lang=TGT_LANG)
            print(json.dumps({"ok": True, "translations": translated}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
