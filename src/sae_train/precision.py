"""The precision seam.

Everything that genuinely differs between train_sae_FP16 / FP8 / FP4 lives
behind this one small interface. The shared runners (runners.py) are written
against `PrecisionPolicy` and call exactly three hooks:

  * add_dtype_args(parser)   — register the precision-specific --dtype / --sae-dtype
                               (their valid choices and defaults differ per precision).
  * build_sae_cfg(...)       — construct the TrainingSAEConfig (fp8/fp4 may need
                               quant config, scaling, or a different SAE class).
  * resolve_autocast(args)   — decide whether the SAE-forward autocast/GradScaler
                               is enabled (precision-dependent kernel support).

Keep precision-specific imports (e.g. transformer_engine for fp8) inside the
concrete policy so importing one entrypoint never drags in another's deps.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser


class PrecisionPolicy(ABC):
    """Base class for a numeric-precision training policy.

    `name` is a short tag (e.g. "fp16") used only for messaging; run-name dtype
    tags come from the resolved --dtype value via utils.DTYPE_SHORT.
    """

    name: str = "base"

    @abstractmethod
    def add_dtype_args(self, parser: ArgumentParser) -> None:
        """Register the precision-specific dtype arguments on `parser`.

        At minimum this should define `--dtype` (activation/buffer dtype) and
        `--sae-dtype` (SAE weight/optimizer dtype) with choices and defaults
        appropriate to this precision.
        """
        ...

    @abstractmethod
    def build_sae_cfg(self, args, d_sae: int, k, training_steps: int):
        """Return a (Training)SAEConfig for the chosen architecture.

        `k` is passed explicitly (not read from args.k) so the k-sweep can vary
        it per SAE while sharing everything else.
        """
        ...

    @abstractmethod
    def resolve_autocast(self, args) -> bool:
        """Return whether the SAE-forward autocast (and SAELens GradScaler) should
        be enabled for this run, given the resolved dtype args."""
        ...
