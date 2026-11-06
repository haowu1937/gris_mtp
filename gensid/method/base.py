import torch
import torch.nn as nn


class SIDMethod(nn.Module):
    #: Name of the Trainer (key into the trainer registry) that drives this method.
    trainer = "base"

    def __init__(self):
        super().__init__()
        self._codebook_sizes = None

    @property
    def n_codebooks(self):
        if self._codebook_sizes is None:
            raise NotImplementedError(
                "Subclasses must set the attribute `_codebook_sizes` used to retrieve the number of codebooks"
            )

        return len(self._codebook_sizes)

    @property
    def codebook_sizes(self):
        if self._codebook_sizes is None:
            raise NotImplementedError(
                "Subclasses must set the attribute `_codebook_sizes` to specify the size of the codebooks"
            )

        return self._codebook_sizes

    @torch.no_grad()
    def generate(self):
        raise NotImplementedError("Subclasses must implement this method to generate SIDs")

    def compute_loss(self):
        raise NotImplementedError("Subclasses must implement this method to compute the loss")
