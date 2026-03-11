#!/usr/bin/env python3

import argparse
from copy import deepcopy
from pathlib import Path

import torch


def build_bayesian_model_string(
    base_model_string,
    alpha_init,
    alpha_min,
    alpha_max,
    kl_weight,
    sample_inference,
    hidden_dim,
):
    if "use_bayesian_decoder=" in base_model_string:
        return base_model_string

    insertion = (
        f", use_bayesian_decoder=True"
        f", bayesian_alpha_init={alpha_init}"
        f", bayesian_alpha_min={alpha_min}"
        f", bayesian_alpha_max={alpha_max}"
        f", bayesian_kl_weight={kl_weight}"
        f", bayesian_sample_inference={sample_inference}"
    )
    if hidden_dim is not None:
        insertion += f", bayesian_hidden_dim={hidden_dim}"

    return base_model_string[:-2] + insertion + "))"


def main():
    parser = argparse.ArgumentParser(
        description="Create a Bayesian-decoder initialization checkpoint from a CUT3R checkpoint."
    )
    parser.add_argument("--input", required=True, help="Path to the source CUT3R checkpoint.")
    parser.add_argument("--output", required=True, help="Path to the output Bayesian checkpoint.")
    parser.add_argument("--alpha-init", type=float, default=0.1)
    parser.add_argument("--alpha-min", type=float, default=1e-6)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-8)
    parser.add_argument("--sample-inference", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # The repo expects these imports to be available when eval'ing the model string.
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf

    base_model_string = ckpt["args"].model
    bayesian_model_string = build_bayesian_model_string(
        base_model_string=base_model_string,
        alpha_init=args.alpha_init,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        kl_weight=args.kl_weight,
        sample_inference=args.sample_inference,
        hidden_dim=args.hidden_dim,
    )

    model = eval(
        bayesian_model_string,
        {
            "ARCroco3DStereo": ARCroco3DStereo,
            "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
            "inf": inf,
        },
    )
    load_result = model.load_state_dict(ckpt["model"], strict=False)

    output_ckpt = deepcopy(ckpt)
    output_ckpt["model"] = model.state_dict()
    output_ckpt["args"] = deepcopy(ckpt["args"])
    output_ckpt["args"].model = bayesian_model_string

    torch.save(output_ckpt, output_path)

    print(f"Input checkpoint:  {input_path}")
    print(f"Output checkpoint: {output_path}")
    print(f"Model string:      {bayesian_model_string}")
    print(f"Missing keys:      {len(load_result.missing_keys)}")
    for key in load_result.missing_keys:
        print(f"  MISSING {key}")
    print(f"Unexpected keys:   {len(load_result.unexpected_keys)}")
    for key in load_result.unexpected_keys:
        print(f"  UNEXPECTED {key}")


if __name__ == "__main__":
    main()
