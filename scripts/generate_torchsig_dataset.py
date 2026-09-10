#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import json
import argparse

import numpy as np
import scipy.io as scio

from torchsig.signals.builders.constellation import all_symbol_maps, constellation_modulator_baseband


from RAG.rag import RAGSearch
from RAG.llm import LLM


DEFAULT_CHUNKS_FOLDER = "./RAG/Knowledge_Base/Chunks"


#ALL AVAILABLE KEYS
# dict_keys(['ook', 'bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk', 
# '4ask', '8ask', '16ask', '32ask', '64ask', 
# '16qam', '32qam', '64qam', '256qam', '1024qam', 
# '32qam_cross', '128qam_cross', '512qam_cross', 
# '16apsk', '32apsk'])

MODULATIONS = [
    "bpsk",
    "qpsk",
    "8psk",
    "16psk",
    "32psk",
    "16qam",
    "32qam",
    "ook",
    "4ask",
    "8ask",
    "16ask",
    "32ask",
]

def bits_to_symbol_indices(bits, bits_per_symbol):

    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) % bits_per_symbol != 0:
        raise ValueError(
            "Number of bits must be divisible by bits_per_symbol"
        )

    bit_groups = bits.reshape(
        -1,
        bits_per_symbol
    )

    symbol_indices = np.zeros(
        len(bit_groups),
        dtype=np.int64
    )

    for i in range(bits_per_symbol):
        symbol_indices = (
            symbol_indices << 1
        ) | bit_groups[:, i]

    return symbol_indices

def generate_bits(num_symbols, bits_per_symbol, rng):

    return rng.integers(
        low=0,
        high=2,
        size=num_symbols * bits_per_symbol,
        dtype=np.uint8,
    )

def get_bits_per_symbol(modulation):

    constellation_size = len(
        all_symbol_maps[modulation]
    )

    return int(
        np.log2(constellation_size)
    )

def build_context(results, max_chars=4000):

    pieces = []

    for r in results:
        pieces.append(
            f"--- Source: {r.get('id', '')} ---\n"
            f"{r.get('text', '')}\n"
        )

    context = "\n".join(pieces)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[truncated]..."

    return context


def get_modulation(metadata):

    return (
        metadata.get("class_name")
        or metadata.get("modulation")
        or metadata.get("signal_type")
        or "unknown"
    )

def get_signal_metadata(signal):
    metadata = {}

    if hasattr(signal, "metadata"):
        metadata.update(signal.metadata)

    for component in getattr(signal, "component_signals", []):
        if hasattr(component, "metadata"):
            metadata.update(component.metadata)

    return metadata

def build_rag_query(modulation, metadata):

    return f"""
    Explain the communication signal {modulation}.
    Describe its modulation, constellation geometry, IQ behavior,
    symbol structure, and BER/SNR characteristics.

    Signal metadata:
    {json.dumps(metadata, default=str)}
    """.strip()


def build_llm_prompt(modulation, context):

    return f"""
    SIGNAL:
    {modulation}

    COMMUNICATION CONTEXT:
    {context}

    Write exactly TWO sentences describing this communication
    signal as a conditioning prompt for a generative AI model.
    """.strip()


def generate_dataset(
    output_dir,
    num_samples,
    signal_length,
    top_k,
    chunks_folder,
    seed,
):

    os.makedirs(output_dir, exist_ok=True)

    rag = RAGSearch(
        chunks_folder=chunks_folder
    )

    llm = LLM()

    rng = np.random.default_rng(seed)

    for idx in range(num_samples):

        modulation = rng.choice(MODULATIONS)

        bits_per_symbol = get_bits_per_symbol(
            modulation
        )

        samples_per_symbol = 1

        num_symbols = signal_length // samples_per_symbol

        bits = generate_bits(
            num_symbols=num_symbols,
            bits_per_symbol=bits_per_symbol,
            rng=rng,
        )

        symbol_indices = bits_to_symbol_indices(
            bits,
            bits_per_symbol,
        )

        iq = constellation_modulator_baseband(
            constellation_name=modulation,
            pulse_shape_name="rectangular",
            max_num_samples=signal_length,
            oversampling_rate_nominal=samples_per_symbol,
            rng=rng,
            symbol_indices=symbol_indices,
        )

        iq = np.asarray(iq).reshape(-1)

        if iq.size < signal_length:
            iq = np.pad(
                iq,
                (0, signal_length - iq.size),
            )

        else:
            iq = iq[:signal_length]

        iq = iq.astype(np.complex64)

        metadata = {
            "modulation": modulation,
            "bits_per_symbol": bits_per_symbol,
            "num_symbols": num_symbols,
            "sample_rate": 1e6,
            "signal_length": signal_length,
        }

        query = build_rag_query(
            modulation,
            metadata,
        )

        results = rag.search(
            query,
            top_k=top_k,
        )

        context = build_context(results)

        prompt = build_llm_prompt(
            modulation,
            context,
        )

        prompt = llm.generate(
            prompt,
            max_new_tokens=196,
            temperature=0.4,
            top_p=0.9,
        )

        sample = {
            "data": iq,
            "bits": bits,
            "symbols": symbol_indices,
            "modulation": np.array(
                [modulation],
                dtype=object,
            ),
            "label": np.array(
                [prompt],
                dtype=object,
            ),
            "metadata": np.array(
                [metadata],
                dtype=object,
            ),
        }

        path = os.path.join(
            output_dir,
            f"sample_{idx:06d}.mat",
        )

        scio.savemat(
            path,
            sample,
        )

        print(
            f"[{idx + 1}/{num_samples}] "
            f"{modulation} | "
            f"{bits_per_symbol} bits/symbol | "
            f"{len(bits)} bits | "
            f"{path}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--signal_length",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--chunks_folder",
        type=str,
        default=DEFAULT_CHUNKS_FOLDER
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        signal_length=args.signal_length,
        top_k=args.top_k,
        chunks_folder=args.chunks_folder,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()