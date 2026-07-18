"""Entry point for CMASD ablation training."""

from knowledge_pretraining.training.ablation_runner import (
    apply_runtime_args,
    main,
    parse_args,
)

if __name__ == "__main__":
    args = parse_args()
    apply_runtime_args(args)
    main()
