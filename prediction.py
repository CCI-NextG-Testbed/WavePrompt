# inference_prompt.py

import csv
import os
from argparse import ArgumentParser

import numpy as np
import scipy.io as scio
import torch
import matplotlib.pyplot as plt

from WavePrompt.stablediff_test.diffusion import SignalDiffusion, GaussianDiffusion
from WavePrompt.stablediff_test.models import tfdiff_Simple
from WavePrompt.stablediff_test.params import AttrDict, params_simple


SUPPORTED_MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]
MOD_TO_ID = {m: i for i, m in enumerate(SUPPORTED_MODS)}
ID_TO_MOD = {i: m for m, i in MOD_TO_ID.items()}


def _normalize_mod_name(txt: str) -> str:
    return str(txt).upper().replace("-", "").replace(" ", "")


def _prompt_respects_modulation(prompt: str, modulation_hint: str) -> bool:
    if not modulation_hint:
        return True
    prompt_norm = _normalize_mod_name(prompt)
    hint_norm = _normalize_mod_name(modulation_hint)
    if hint_norm not in prompt_norm:
        return False
    for mod in SUPPORTED_MODS:
        mod_norm = _normalize_mod_name(mod)
        if mod_norm != hint_norm and mod_norm in prompt_norm:
            return False
    return True


def infer_modulation_from_text(text: str) -> str:
    text_norm = _normalize_mod_name(text)
    for mod in SUPPORTED_MODS:
        if _normalize_mod_name(mod) in text_norm:
            return mod
    return ""


def _mod_order(modulation: str) -> int:
    m = _normalize_mod_name(modulation)
    if m == "BPSK":
        return 2
    if m == "QPSK":
        return 4
    if m == "8PSK":
        return 8
    if m == "16QAM":
        return 16
    if m == "64QAM":
        return 64
    if m == "256QAM":
        return 256
    return 2


def _bits_per_symbol(modulation: str) -> int:
    M = _mod_order(modulation)
    return int(np.log2(M))


def _bits_to_symbol_index(bits: np.ndarray, k: int) -> np.ndarray:
    bits = np.asarray(bits).reshape(-1)
    if k <= 0 or bits.size < k:
        return np.zeros((0,), dtype=np.float32)
    T = bits.size // k
    bb = bits[: T * k].reshape(T, k).astype(np.int64)
    w = (2 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return (bb * w[None, :]).sum(axis=1).astype(np.float32)


def mat_to_prompt_str(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")

    arr = np.asarray(v)
    if arr.dtype == object:
        if arr.size == 0:
            return ""
        return mat_to_prompt_str(arr.ravel()[0])

    if arr.dtype.kind in ("U", "S"):
        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return "".join(arr.ravel().tolist())
        if arr.size == 1:
            return str(arr.item())
        return "".join(arr.astype(str).ravel().tolist())

    if arr.size == 1:
        return str(arr.item())
    return str(arr)


def load_cond_from_mat(mat_path: str, prompt_key="prompt", bits_key="bits"):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = scio.loadmat(mat_path, verify_compressed_data_integrity=False)
    if prompt_key not in mat:
        raise KeyError(f"Missing '{prompt_key}' in {mat_path}. Keys: {sorted(mat.keys())}")
    if bits_key not in mat:
        raise KeyError(f"Missing '{bits_key}' in {mat_path}. Keys: {sorted(mat.keys())}")

    prompt = mat_to_prompt_str(mat[prompt_key])
    bits = np.asarray(mat[bits_key]).squeeze().astype(np.uint8, copy=False)
    if bits.ndim != 1:
        bits = bits.reshape(-1).astype(np.uint8, copy=False)
    return prompt, bits


def build_bits_cond(bits: np.ndarray, N: int, modulation: str = "", sps: int = 1) -> np.ndarray:
    bits = np.asarray(bits).reshape(-1)
    bits = (bits != 0).astype(np.float32)
    if bits.size == 0:
        return np.zeros((N,), dtype=np.float32)

    modulation = str(modulation or "").upper()
    if modulation in SUPPORTED_MODS:
        k = _bits_per_symbol(modulation)
        M = _mod_order(modulation)
        sym_idx = _bits_to_symbol_index(bits, k)
        if sym_idx.size == 0:
            return np.zeros((N,), dtype=np.float32)
        if M > 1:
            sym_idx = sym_idx / float(M - 1)
        bits_cond = np.repeat(sym_idx, max(1, int(sps))).astype(np.float32)
        if bits_cond.size < N:
            bits_cond = np.pad(bits_cond, (0, N - bits_cond.size), mode="constant")
        elif bits_cond.size > N:
            bits_cond = bits_cond[:N]
        return bits_cond

    if bits.size == N:
        return bits.astype(np.float32, copy=False)
    idx = np.floor(np.linspace(0, bits.size - 1, N)).astype(np.int64)
    return bits[idx].astype(np.float32, copy=False)


def save_mat(path, iq_tensor, prompt, bits):
    x = iq_tensor[0]
    x_complex = torch.view_as_complex(x)
    mat = {
        "iq": x_complex.cpu().numpy(),
        "prompt": np.array([prompt], dtype=object),
        "bits": np.asarray(bits, dtype=np.uint8),
    }
    scio.savemat(path, mat)


def get_model_and_diffusion(args):
    params = params_simple
    model_dir = args.model_dir or params.model_dir

    if args.cond_dir is not None:
        params.cond_dir = args.cond_dir

    device = torch.device("cpu")
    weights_path = os.path.join(model_dir, "weights.pt")
    checkpoint = torch.load(weights_path, map_location=device) if os.path.exists(weights_path) else torch.load(model_dir, map_location=device)

    model = tfdiff_Simple(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.params.override(params)

    diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
    return params, model, diffusion, device


def maybe_load_llm(enable_llm: bool):
    if not enable_llm:
        return None
    try:
        from RAG.llm import LLM  # optional
        return LLM()
    except Exception:
        return None


def maybe_load_rag(enable_rag: bool, chunks_folder: str):
    if not enable_rag:
        return None, None
    try:
        from RAG.rag import RAGSearch, build_context
        rag = RAGSearch(chunks_folder=chunks_folder)
        return rag, build_context
    except Exception:
        return None, None


def maybe_load_similarity_model(enable_similarity: bool):
    if not enable_similarity:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def semantic_similarity(text_a: str, text_b: str, model) -> float:
    if model is None:
        return float("nan")
    emb = model.encode([str(text_a), str(text_b)], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def load_prompt_lines(path: str):
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


def _signal_params_dict(args, params, modulation_hint: str = "") -> dict:
    return {
        "modulation": modulation_hint or "",
        "number_of_samples": int(args.num_samples) if args.num_samples is not None else int(params.sample_rate),
        "sampling_rate_hz": float(args.sampling_freq_hz),
        "amplitude": float(args.amplitude),
        "center_frequency_hz": float(args.center_freq_hz),
        "phase_degrees": float(args.phase_deg),
        "samples_per_symbol": int(args.samples_per_symbol),
    }


def _signal_params_text(signal_params: dict) -> str:
    return (
        f"Number of samples: {signal_params['number_of_samples']}; "
        f"Sampling rate: {signal_params['sampling_rate_hz']} Hz; "
        f"Amplitude: {signal_params['amplitude']}; "
        f"Center frequency: {signal_params['center_frequency_hz']} Hz; "
        f"Initial phase: {signal_params['phase_degrees']} degrees; "
        f"Samples per symbol: {signal_params['samples_per_symbol']}."
    )


def _signal_config_line(signal_params: dict) -> str:
    return (
        f"Modulation={signal_params['modulation']}; "
        f"N={signal_params['number_of_samples']}; "
        f"sampling_rate_hz={signal_params['sampling_rate_hz']}; "
        f"samples_per_symbol={signal_params['samples_per_symbol']}; "
        f"amplitude={signal_params['amplitude']}; "
        f"center_frequency_hz={signal_params['center_frequency_hz']}; "
        f"initial_phase_deg={signal_params['phase_degrees']}."
    )


def build_rag_query(style: str, modulation_hint: str = "", signal_params: dict = None) -> str:
    style_hint = {
        "dumb": "simple beginner explanation",
        "vague": "high-level robust communication guidance",
        "advanced": "technical symbol-level and BER/SNR explanation",
    }.get(style, "general explanation")
    params_text = _signal_params_text(signal_params) if signal_params is not None else ""
    if modulation_hint:
        return (
            f"Digital modulation in complex baseband IQ for {modulation_hint}. "
            f"Need {style_hint}. Restrict the discussion to {modulation_hint} only. "
            f"Include {modulation_hint} constellation behavior, symbol mapping, "
            f"time-domain appearance, and BER/SNR tradeoffs. "
            f"Use these authoritative signal parameters: {params_text} "
            f"Do not discuss other modulations or invent different parameter values."
        )
    return (
        "Digital modulation in complex baseband IQ. "
        f"Need {style_hint}. Include constellation behavior, symbol mapping, "
        f"time-domain appearance, and BER/SNR tradeoffs. "
        f"Use these authoritative signal parameters: {params_text}"
    )


def generate_prompt(style: str, llm=None, rag_context: str = "", modulation_hint: str = "", signal_params: dict = None) -> str:
    style = style.lower().strip()
    params_text = _signal_params_text(signal_params) if signal_params is not None else ""

    def fallback_prompt() -> str:
        if style == "dumb":
            return "Write me a short poem about a potato."
        if style == "vague":
            if modulation_hint:
                return f"Generate a robust low-SNR waveform using exactly {modulation_hint}. Use exactly these parameters: {params_text}"
            return "generate a robust low-snr waveform"
        if style == "advanced":
            base = (
                "Generate a complex baseband waveform with clear symbol separability, "
                "stable phase behavior, and demodulation-friendly structure."
            )
            if modulation_hint:
                base += f" Use exactly {modulation_hint} modulation."
            if params_text:
                base += f" Use exactly these parameters: {params_text}"
            return base
        if modulation_hint:
            return f"Generate a communication signal using exactly {modulation_hint}. Use exactly these parameters: {params_text}"
        return "Generate a communication signal."

    if llm is not None:
        style_desc = {
            "dumb": "very short and basic",
            "vague": "ambiguous and vague",
            "advanced": "technical and detailed",
        }.get(style, "mixed clarity")
        if style == "dumb":
            prompt = (
                "Write one short user prompt that is unrelated to communications, signals, radio, "
                "modulation, BER, or electronics. One sentence only."
            )
        else:
            if rag_context:
                prompt = (
                    f"Write one short user prompt to request generation of a communication waveform.\n"
                    f"Style: {style_desc}. "
                    f"Required modulation: {modulation_hint if modulation_hint else 'not specified'}. "
                    f"Authoritative signal parameters: {params_text} "
                    f"If a modulation is specified, you must use exactly that modulation name and must not substitute any other modulation. "
                    f"You must use the given numerical values exactly and must not invent alternate sample counts, sampling rates, center frequencies, amplitudes, phases, or samples-per-symbol. "
                    f"One sentence only.\n"
                    f"Use only concepts from this context:\n{rag_context}"
                )
            else:
                prompt = (
                    f"Write one short user prompt to request generation of a communication waveform. "
                    f"Style: {style_desc}. "
                    f"Required modulation: {modulation_hint if modulation_hint else 'not specified'}. "
                    f"Authoritative signal parameters: {params_text} "
                    f"If a modulation is specified, you must use exactly that modulation name and must not substitute any other modulation. "
                    f"You must use the given numerical values exactly and must not invent alternate sample counts, sampling rates, center frequencies, amplitudes, phases, or samples-per-symbol. "
                    f"One sentence only."
                )
        try:
            out = llm.generate(prompt, max_new_tokens=96, temperature=0.75, top_p=0.9)
            txt = str(out).strip()
            if txt and _prompt_respects_modulation(txt, modulation_hint):
                return txt
        except Exception:
            pass

    return fallback_prompt()


def _semantic_similarity_band(idx: int, steps: int):
    if steps <= 2:
        return (0.95, 1.00)
    if idx <= 0:
        return (-1.0, 0.10)
    if idx >= steps - 1:
        return (0.98, 1.00)
    explicit_bands = [
        (0.10, 0.22),
        (0.22, 0.34),
        (0.34, 0.46),
        (0.46, 0.58),
        (0.58, 0.70),
        (0.70, 0.80),
        (0.80, 0.88),
        (0.88, 0.94),
    ]
    if steps == 10 and 1 <= idx <= 8:
        return explicit_bands[idx - 1]
    frac = (idx - 1) / float(max(1, steps - 2))
    center = 0.16 + 0.74 * frac
    half_width = 0.07
    lo = max(0.08, center - half_width)
    hi = min(0.96, center + half_width)
    return (lo, hi)


def _fallback_semantic_prompt(base_prompt: str, modulation_hint: str, signal_params: dict, idx: int, steps: int) -> str:
    params_text = _signal_params_text(signal_params) if signal_params is not None else ""
    mod_text = f"{modulation_hint} " if modulation_hint else ""
    templates = [
        "Create a digital communication waveform.",
        f"Generate a {mod_text}communication signal with the provided bits.",
        f"Generate a {mod_text}baseband signal suitable for transmission.",
        f"Generate a {mod_text}signal using the provided bits and these parameters: {params_text}",
        f"Generate a {mod_text}waveform with a recognizable constellation and stable symbol behavior.",
        f"Generate a {mod_text}baseband waveform with clear symbol separability, stable phase behavior, and these parameters: {params_text}",
        f"Generate a {mod_text}complex baseband waveform consistent with this request: {base_prompt}",
    ]
    pick = min(len(templates) - 1, max(0, int(round((idx / float(max(1, steps - 1))) * (len(templates) - 1)))))
    return templates[pick]


def _prompt_duplicate_penalty(candidate: str, existing_prompts, sim_model) -> float:
    if sim_model is None or not existing_prompts:
        return 0.0
    sims = [semantic_similarity(candidate, prior, sim_model) for prior in existing_prompts]
    if not sims:
        return 0.0
    return max(sims)


def generate_semantic_prompt_ladder(base_prompt: str, llm, rag_context: str, modulation_hint: str, signal_params: dict, steps: int, sim_model):
    steps = max(2, int(steps))
    params_text = _signal_params_text(signal_params) if signal_params is not None else ""
    config_line = _signal_config_line(signal_params) if signal_params is not None else ""
    prompts = []

    for idx in range(steps):
        if idx == 0:
            prompts.append("What is the capital of Brazil?")
            continue
        if idx == steps - 1:
            prompts.append(str(base_prompt).strip())
            continue

        target_lo, target_hi = _semantic_similarity_band(idx, steps)
        target_mid = 0.5 * (target_lo + target_hi)
        best_text = _fallback_semantic_prompt(base_prompt, modulation_hint, signal_params, idx, steps)
        best_score = semantic_similarity(base_prompt, best_text, sim_model) if sim_model is not None else float("nan")
        best_gap = abs(best_score - target_mid) if sim_model is not None else float("inf")
        previous_scores = [semantic_similarity(base_prompt, prior, sim_model) for prior in prompts] if sim_model is not None else []
        prev_score = previous_scores[-1] if previous_scores else -1.0

        candidates = [best_text]
        if llm is not None:
            previous_prompt_text = "\n".join([f"- {p}" for p in prompts]) if prompts else "(none)"
            for attempt in range(10):
                temperature = min(0.95, 0.55 + 0.07 * attempt)
                prompt = (
                    f"SIGNAL CONFIG (authoritative; use exactly these values):\n"
                    f"{config_line}\n\n"
                    f"BASE PROMPT:\n{base_prompt}\n\n"
                    f"ALREADY USED PROMPTS:\n{previous_prompt_text}\n\n"
                    f"CONTEXT (RAG output; ONLY use facts stated here; do not invent):\n"
                    f"{rag_context}\n\n"
                    f"TARGET MODULATION: {modulation_hint if modulation_hint else 'not specified'}\n"
                    f"AUTHORITATIVE SIGNAL PARAMETERS: {params_text}\n"
                    f"Generate exactly one sentence.\n"
                    f"The sentence must target cosine similarity in the range [{target_lo:.2f}, {target_hi:.2f}] "
                    f"relative to the base prompt.\n"
                    f"It must be meaningfully different from the already used prompts.\n"
                    f"Its cosine similarity must be greater than the previous rung and less than the next higher bands.\n"
                    f"For lower similarity, use broader or more indirect wording while staying in waveform generation.\n"
                    f"For higher similarity, stay closer to the original meaning and wording.\n"
                    f"If modulation is specified, use exactly that modulation name and do not mention any other modulation."
                )
                try:
                    out = llm.generate(prompt, max_new_tokens=96, temperature=temperature, top_p=0.9)
                    txt = str(out).strip()
                    if txt:
                        candidates.append(txt)
                except Exception:
                    pass

        for txt in candidates:
            if not txt or not _prompt_respects_modulation(txt, modulation_hint):
                continue
            score = semantic_similarity(base_prompt, txt, sim_model)
            duplicate_penalty = _prompt_duplicate_penalty(txt, prompts, sim_model)
            in_band = sim_model is not None and target_lo <= score <= target_hi
            monotonic_ok = sim_model is None or score > prev_score + 0.03
            unique_ok = duplicate_penalty < 0.97
            if in_band and monotonic_ok and unique_ok:
                best_text = txt
                best_score = score
                best_gap = 0.0
                break
            gap = abs(score - target_mid) if sim_model is not None else float("inf")
            adjusted_gap = gap + max(0.0, duplicate_penalty - 0.92) * 5.0
            if sim_model is not None and score <= prev_score:
                adjusted_gap += (prev_score - score + 0.03) * 3.0
            if adjusted_gap < best_gap:
                best_text = txt
                best_score = score
                best_gap = adjusted_gap

        prompts.append(best_text)

    return prompts


def infer_router_stats(model):
    model_ref = model.module if hasattr(model, "module") else model
    probs = getattr(model_ref, "last_mod_probs", None)
    if probs is None:
        return "NA", float("nan")
    idx = int(torch.argmax(probs[0]).item())
    conf = float(probs[0, idx].item())
    inv = {v: k for k, v in model_ref.mod_to_id.items()}
    return inv.get(idx, "NA"), conf


def _default_rfml_labels():
    # Common RML2016.10a class order used in many RFML examples.
    return ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]


def maybe_load_rfml_classifier(path: str, input_samples: int, n_classes: int, require_rfml: bool = False):
    if not path:
        return None
    if not os.path.exists(path):
        msg = f"RFML classifier checkpoint not found: {path}"
        if require_rfml:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}")
        return None
    try:
        from rfml.nn.model import build_model
    except Exception as e:
        msg = f"rfml package unavailable: {e}"
        if require_rfml:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
        return None

    model = build_model(model_name="CNN", input_samples=int(input_samples), n_classes=int(n_classes))
    try:
        # Prefer rfml model API if available.
        if hasattr(model, "load"):
            model.load(path)
        else:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            model.load_state_dict(state, strict=False)
        model.eval()
        print(f"Loaded RFML classifier: {path}")
        return model
    except Exception as e:
        msg = f"failed loading RFML classifier weights: {e}"
        if require_rfml:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
        return None


def rfml_predict(pred_tensor: torch.Tensor, rfml_model, rfml_input_samples: int, rfml_labels):
    """
    pred_tensor: [1, N, 1, 2]
    Returns (label, confidence)
    """
    if rfml_model is None:
        return "NA", float("nan")
    try:
        x = pred_tensor[:, :, 0, :]  # [1,N,2]
        x = x.permute(0, 2, 1).contiguous().to(torch.float32)  # [1,2,N] = [B,IQ,T]
        N = x.shape[-1]
        M = int(rfml_input_samples)
        if N != M:
            idx = torch.linspace(0, N - 1, steps=M, device=x.device).long()
            x = x.index_select(dim=-1, index=idx)

        # RFML CNN expects 4D: [B, C, IQ, T].
        # Use a single channel C=1 for complex IQ streams.
        x = x.unsqueeze(1)  # [B,1,IQ,T]

        with torch.no_grad():
            logits = rfml_model(x)
            probs = torch.softmax(logits, dim=-1)
            k = int(torch.argmax(probs[0]).item())
            conf = float(probs[0, k].item())
        label = rfml_labels[k] if 0 <= k < len(rfml_labels) else f"class_{k}"
        return label, conf
    except Exception as e:
        print(f"[warn] RFML predict failed: {e}")
        return "NA", float("nan")


def run_single(args):
    params, model, diffusion, device = get_model_and_diffusion(args)
    out_path = args.out_dir or params.out_dir

    user_prompt, bits = load_cond_from_mat(args.file, prompt_key="prompt", bits_key="bits")
    modulation_hint = infer_modulation_from_text(user_prompt)
    bits_cond = build_bits_cond(
        bits,
        N=int(params.sample_rate),
        modulation=modulation_hint,
        sps=int(args.samples_per_symbol),
    )

    with torch.no_grad():
        cond = {"prompt": prompt, "bits": bits}  # bits is np.uint8 1D vector
        pred = diffusion.sampling(model, cond, device)

    print(f"Saving to {out_path}")
    save_mat(out_path, pred, prompt, bits)


def run_batch(args):
    params, model, diffusion, device = get_model_and_diffusion(args)
    rng = np.random.default_rng(args.seed)

    out_dir = args.batch_out_dir or args.out_dir or "./results/prediction_batch"
    os.makedirs(out_dir, exist_ok=True)

    llm = maybe_load_llm(args.use_llm_prompt)
    rag, rag_build_context = maybe_load_rag(args.use_rag, args.chunks_folder)
    sim_model = maybe_load_similarity_model(bool(args.semantic_base_prompt))
    rfml_labels = [s.strip() for s in args.rfml_label_map.split(",")] if args.rfml_label_map.strip() else _default_rfml_labels()
    rfml_clf = maybe_load_rfml_classifier(
        path=args.rfml_clf_path,
        input_samples=int(args.rfml_input_samples),
        n_classes=len(rfml_labels),
        require_rfml=bool(args.require_rfml),
    )
    if args.require_rfml and rfml_clf is None:
        raise RuntimeError("RFML classifier is required but could not be loaded.")

    # Build either a semantic ladder or a balanced mixed schedule.
    semantic_mode = bool(args.semantic_base_prompt)
    if semantic_mode:
        base_signal_params = _signal_params_dict(args, params, modulation_hint=str(args.semantic_modulation).upper())
        if args.semantic_prompts_file:
            prompt_schedule = load_prompt_lines(args.semantic_prompts_file)
            n = len(prompt_schedule)
            style_schedule = ["semantic_custom" for _ in range(n)]
            mod_schedule = [str(args.semantic_modulation).upper() for _ in range(n)]
        else:
            n = int(args.semantic_steps)
            style_schedule = ["semantic" for _ in range(n)]
            mod_schedule = [str(args.semantic_modulation).upper() for _ in range(n)]
            rag_ctx = ""
            if rag is not None and rag_build_context is not None:
                try:
                    q = build_rag_query(style="advanced", modulation_hint=str(args.semantic_modulation).upper(), signal_params=base_signal_params)
                    rag_results = rag.search(q, top_k=int(args.rag_top_k))
                    rag_ctx = rag_build_context(rag_results, max_chars=int(args.rag_max_chars))
                except Exception:
                    rag_ctx = ""
            prompt_schedule = generate_semantic_prompt_ladder(
                base_prompt=args.semantic_base_prompt,
                llm=llm,
                rag_context=rag_ctx,
                modulation_hint=str(args.semantic_modulation).upper(),
                signal_params=base_signal_params,
                steps=n,
                sim_model=sim_model,
            )
        shared_bits = rng.integers(0, 2, size=int(args.bits_len), dtype=np.uint8)
    else:
        n = int(args.batch_tests)
        if args.prompt_style == "mixed":
            base_styles = ["dumb", "vague", "advanced"]
            style_schedule = [base_styles[i % len(base_styles)] for i in range(n)]
            rng.shuffle(style_schedule)
        else:
            style_schedule = [args.prompt_style for _ in range(n)]

        mod_schedule = ["" for _ in range(n)]
        if args.random_modulation_hint:
            mod_base = ["BPSK", "QPSK", "8PSK", "16QAM"]
            nondumb_idx = [i for i, s in enumerate(style_schedule) if s != "dumb"]
            nondumb_mods = [mod_base[i % len(mod_base)] for i in range(len(nondumb_idx))]
            rng.shuffle(nondumb_mods)
            for i, m in zip(nondumb_idx, nondumb_mods):
                mod_schedule[i] = m
        prompt_schedule = None
        shared_bits = None

    rows = []
    for i in range(n):
        style = style_schedule[i]
        modulation_hint = mod_schedule[i]
        signal_params = _signal_params_dict(args, params, modulation_hint=modulation_hint)

        if semantic_mode:
            rag_ctx = ""
            prompt = prompt_schedule[i]
            bits = shared_bits.copy()
            semantic_score = semantic_similarity(args.semantic_base_prompt, prompt, sim_model)
        else:
            rag_ctx = ""
            if rag is not None and rag_build_context is not None:
                try:
                    q = build_rag_query(style=style, modulation_hint=modulation_hint, signal_params=signal_params)
                    rag_results = rag.search(q, top_k=int(args.rag_top_k))
                    rag_ctx = rag_build_context(rag_results, max_chars=int(args.rag_max_chars))
                except Exception:
                    rag_ctx = ""

            prompt = generate_prompt(
                style=style,
                llm=llm,
                rag_context=rag_ctx,
                modulation_hint=modulation_hint,
                signal_params=signal_params,
            )
            bits = rng.integers(0, 2, size=int(args.bits_len), dtype=np.uint8)
            semantic_score = float("nan")
        cosine_similarity_score = semantic_score
        bits_cond = build_bits_cond(
            bits,
            N=int(params.sample_rate),
            modulation=modulation_hint,
            sps=int(args.samples_per_symbol),
        )

        with torch.no_grad():
            cond = {"prompt": prompt, "bits_cond": bits_cond}
            pred = diffusion.sampling(model, cond, device)

        if args.plot:
            # pred shape: [B, N, 1, 2] where last dim is [I, Q]
            iq = pred[0, :, 0, :].detach().cpu().numpy()  # [N,2]
            i_data = iq[:, 0]
            q_data = iq[:, 1]

            # Optional downsample for cleaner scatter and faster plotting.
            step = max(1, int(args.plot_stride))
            i_plot = i_data[::step]
            q_plot = q_data[::step]

            plt.figure(figsize=(6, 6))
            plt.plot(i_plot, q_plot, ".", markersize=2)
            req_mod = modulation_hint if modulation_hint else "NA"
            title_prompt = prompt if len(prompt) <= 120 else (prompt[:117] + "...")
            plt.title(f"Req Mod: {req_mod}\nPrompt: {title_prompt}")
            plt.xlabel("In-Phase (I)")
            plt.ylabel("Quadrature (Q)")
            plt.grid(True, alpha=0.4)
            plt.axhline(0, color="black", linewidth=0.5)
            plt.axvline(0, color="black", linewidth=0.5)
            plt.axis("equal")
            plot_path = os.path.join(out_dir, f"pred_test_{i:04d}_constellation.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()

        rfml_mod, rfml_conf = rfml_predict(pred, rfml_clf, int(args.rfml_input_samples), rfml_labels)
        requested_mod = modulation_hint if modulation_hint else "NA"
        rfml_match_requested = -1
        if requested_mod != "NA" and rfml_mod != "NA":
            rfml_match_requested = int(str(rfml_mod).upper() == str(requested_mod).upper())

        out_file = os.path.join(out_dir, f"pred_test_{i:04d}.mat")
        save_mat(out_file, pred, prompt, bits)

        rows.append(
            {
                "test_index": i,
                "file": os.path.basename(out_file),
                "labels_per_sample_tag": int(args.labels_per_sample_tag),
                "prompt_style": style,
                "modulation_hint": modulation_hint,
                "requested_modulation": requested_mod,
                "rfml_modulation": rfml_mod,
                "rfml_confidence": rfml_conf,
                "rfml_match_requested": rfml_match_requested,
                "rag_used": 1 if rag_ctx else 0,
                "bits_len": int(args.bits_len),
                "prompt_text": prompt,
                "semantic_score": semantic_score,
                "num_samples": int(signal_params["number_of_samples"]),
                "sampling_freq_hz": float(signal_params["sampling_rate_hz"]),
                "amplitude": float(signal_params["amplitude"]),
                "center_freq_hz": float(signal_params["center_frequency_hz"]),
                "phase_deg": float(signal_params["phase_degrees"]),
                "samples_per_symbol": int(signal_params["samples_per_symbol"]),
            }
        )
        if not np.isnan(cosine_similarity_score):
            print(
                f"[{i+1}/{n}] cosine_similarity_score="
                f"{cosine_similarity_score:.4f} saved {out_file}"
            )
        else:
            print(f"[{i+1}/{n}] saved {out_file}")

    csv_path = os.path.join(out_dir, args.batch_csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "test_index",
                "file",
                "labels_per_sample_tag",
                "prompt_style",
                "modulation_hint",
                "requested_modulation",
                "rfml_modulation",
                "rfml_confidence",
                "rfml_match_requested",
                "rag_used",
                "bits_len",
                "prompt_text",
                "semantic_score",
                "num_samples",
                "sampling_freq_hz",
                "amplitude",
                "center_freq_hz",
                "phase_deg",
                "samples_per_symbol",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved batch CSV: {csv_path}")
    if len(rows) > 0:
        rf_conf = [r["rfml_confidence"] for r in rows if np.isfinite(r["rfml_confidence"])]
        if len(rf_conf) > 0:
            print(f"RFML confidence mean/std: {np.mean(rf_conf):.4f}/{np.std(rf_conf):.4f}")


def main(args):
    if args.batch_tests and args.batch_tests > 0:
        run_batch(args)
    else:
        if not args.file:
            raise ValueError("--file is required for single prediction mode.")
        run_single(args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run prediction (single) or integrated prompt-control batch tests.")

    parser.add_argument("--file", default="", help="MAT file containing 'prompt' and 'bits' (single mode).")
    parser.add_argument("--model_dir", default="./model/simple", help="Directory storing model checkpoints.")
    parser.add_argument("--out_dir", default="./dataset/simple/output/prediction.mat", help="Single-mode output .mat path.")
    parser.add_argument("--cond_dir", default=None, help="Condition directory override.")
    parser.add_argument("--device", default="cuda", help="Reserved arg; script currently runs on CPU.")
    parser.add_argument("--chunks_folder", type=str, default="./RAG/Knowledge_Base/Chunks", help="Unused in prediction path.")

    # Integrated batch mode.
    parser.add_argument("--batch_tests", type=int, default=0, help="Number of integrated random tests to run.")
    parser.add_argument("--batch_out_dir", type=str, default="./results/prediction_batch", help="Batch outputs directory.")
    parser.add_argument("--batch_csv", type=str, default="batch_metrics.csv", help="Batch metrics CSV filename.")
    parser.add_argument("--bits_len", type=int, default=1024, help="Random bitstream length per batch test.")
    parser.add_argument("--num_samples", type=int, default=2048, help="Number of IQ samples to describe in generated batch prompts; defaults to params.sample_rate.")
    parser.add_argument("--sampling_freq_hz", type=float, default=1e6, help="Authoritative sampling frequency to include in generated batch prompts.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Authoritative amplitude to include in generated batch prompts.")
    parser.add_argument("--center_freq_hz", type=float, default=0.0, help="Authoritative center frequency to include in generated batch prompts.")
    parser.add_argument("--phase_deg", type=float, default=0.0, help="Authoritative phase in degrees to include in generated batch prompts.")
    parser.add_argument("--samples_per_symbol", type=int, default=1, help="Samples per symbol used when building modulation-aware bits_cond and batch prompts.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for batch tests.")
    parser.add_argument("--prompt_style", type=str, default="mixed", choices=["dumb", "vague", "advanced", "mixed"], help="Prompt style for batch generation.")
    parser.add_argument("--use_llm_prompt", action="store_true", help="Use local RAG.llm LLM for prompt generation if available.")
    parser.add_argument("--use_rag", action="store_true", help="Use retrieval-augmented context for batch prompt generation.")
    parser.add_argument("--rag_top_k", type=int, default=5, help="Top-k retrieved chunks when --use_rag is enabled.")
    parser.add_argument("--rag_max_chars", type=int, default=1800, help="Max chars of retrieved context fed into prompt generator.")
    parser.add_argument("--random_modulation_hint", action="store_true", help="Include a random modulation hint in each generated prompt.")
    parser.add_argument("--labels_per_sample_tag", type=int, default=1, help="Tag written to batch CSV for grouping runs by training labels-per-sample.")
    parser.add_argument("--rfml_clf_path", type=str, default="./rfml_model/cnn.pt", help="Path to RFML CNN checkpoint (optional).")
    parser.add_argument("--rfml_input_samples", type=int, default=128, help="RFML classifier input sample length.")
    parser.add_argument("--rfml_label_map", type=str, default="", help="Comma-separated RFML class names in index order.")
    parser.add_argument("--require_rfml", action="store_true", help="Fail fast if RFML package/checkpoint/model loading fails.")
    parser.add_argument("--plot", action="store_true", help="Save constellation plot PNG per generated batch sample.")
    parser.add_argument("--plot_stride", type=int, default=1, help="Subsample factor for constellation plotting (e.g., 2 keeps every 2nd point).")
    parser.add_argument("--semantic_base_prompt", type=str, default="", help="If set, run a semantic-similarity ladder around this base prompt.")
    parser.add_argument("--semantic_prompts_file", type=str, default="", help="Optional text file with one custom semantic prompt per non-empty line.")
    parser.add_argument("--semantic_steps", type=int, default=10, help="Number of prompts in semantic ladder mode.")
    parser.add_argument("--semantic_modulation", type=str, default="16QAM", help="Fixed modulation used in semantic ladder mode.")
    main(parser.parse_args())
