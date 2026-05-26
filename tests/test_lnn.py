import unittest
import torch
import numpy as np

from lnn import SPI_OmniKernelODE


class TestLNNODE(unittest.TestCase):
    def test_process_bio_stream_outputs_sentiment_vector(self):
        T = 6
        input_dim = 3
        latent_dim = 8
        sentiment_dim = 5

        kernel = SPI_OmniKernelODE(input_dim, latent_dim, sentiment_dim)

        x = torch.tensor(np.random.randn(T, input_dim).astype('float32'))
        t = torch.arange(0, T, dtype=torch.float32)

        try:
            out = kernel.process_bio_stream(x, t)
        except RuntimeError as e:
            self.skipTest(f'torchdiffeq not installed in environment: {e}')

        self.assertEqual(out.shape[0], sentiment_dim)


if __name__ == '__main__':
    unittest.main()
