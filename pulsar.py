from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
import numpy as np
import tqdm
from diffusers import DDIMScheduler, DDPMScheduler
import torch
from diffusers import UNet2DModel
import os
from bitarray import bitarray
from typing import *
import time
import png

from prg import *
from coding import *

# Suppress FutureWarning
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


REPOS = [
    "google/ddpm-church-256",
    "google/ddpm-celebahq-256",
    "google/ddpm-bedroom-256",
    "google/ddpm-cat-256",
]

DEFAULT_KEY = b"E" * 64
DEFAULT_SEED = b"seed"


class Pulsar:
    def __init__(
        self,
        ecc: Union[ConcatenatedCode, None] = None,
        repo=REPOS[0],
        key=DEFAULT_KEY,
        seed=DEFAULT_SEED,
        scheduler=DDIMScheduler,
        num_inference_steps=50,
        distance="abs",
        benchmarks=True,
    ) -> None:
        # State for benchmarks
        self.benchmarks = {} if benchmarks else None

        self._time_benchmark("__init__")
        self._time_benchmark("model_loading")

        # Check if hardware acceleration is available
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        # Disable Autograd
        torch.set_grad_enabled(False)

        # Load model
        self.model = UNet2DModel.from_pretrained(repo)
        self.model.to(self.device)

        # Load scheduler
        self.scheduler = scheduler.from_config(repo)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        self._time_benchmark("model_loading")

        # Set random seed
        self.prg_key = key
        self.prg_seed = seed
        self._reload_randomness()

        # Error-correcting code
        self.ecc = ecc

        # Distance algorithm
        self.distance = distance

        # Model state required to regenerate for different messages
        self.generate_state = {}

        self._time_benchmark("__init__")

    def save_sample(self, sample, fname, dtype=np.uint16):
        image_processed = sample.cpu().permute(0, 2, 3, 1).numpy()
        dtype_max = np.iinfo(dtype).max

        y = image_processed[0]
        #   z = (dtype_max*((y - y.min())/y.ptp())).astype(dtype)
        z = (y + 1.0) * (dtype_max / 2)
        image_processed = z.astype(dtype)

        # https://stackoverflow.com/a/25814423
        with open(fname, "wb") as f:
            writer = png.Writer(
                width=image_processed.shape[1],
                height=image_processed.shape[0],
                bitdepth=int(np.log2(dtype_max)) + 1,
                greyscale=False,
            )
            image2list = image_processed.reshape(
                -1, image_processed.shape[1] * image_processed.shape[2]
            ).tolist()
            writer.write(f, image2list)

    def load_sample(self, fname, dtype=np.uint16):
        with open(fname, "rb") as f:
            reader = png.Reader(f)
            png_data = reader.asDirect()
            # https://pypng.readthedocs.io/en/latest/ex.html#png-to-numpy-array-reading
            image = np.vstack(list(map(dtype, png_data[2])))

        dtype_max = np.iinfo(dtype).max

        image = image.reshape((1, 256, 256, 3))
        image = image.astype(np.float32)
        image = torch.tensor(image)
        image = (image / (dtype_max / 2)) - 1.0

        sample = image.permute(0, 3, 1, 2)
        sample = sample.to(self.device)
        return sample

    def _time_benchmark(self, name):
        if self.benchmarks is None:
            # Benchmarks are disabled
            return

        if name not in self.benchmarks:
            self.benchmarks[name] = [time.time(), None]
        else:
            if self.benchmarks[name][1] is not None:
                # Benchmark function called more than twice
                raise ValueError()

            self.benchmarks[name][1] = time.time()

    def _multi_benchmark(self, name, benchmark):
        if self.benchmarks is None:
            # Benchmarks are disabled
            return

        if name not in self.benchmarks:
            self.benchmarks[name] = []

        self.benchmarks[name].append(benchmark)

    def time_benchmark_results(self, name):
        return self.benchmarks[name][1] - self.benchmarks[name][0]

    def multi_benchmark_results(self, name):
        return self.benchmarks[name]

    def _reload_randomness(self):
        self.prg = Prg(self.prg_key, self.prg_seed)
        random_bits = self.prg.get_bits(32)
        random_seed = random_bits.dot(1 << np.arange(random_bits.size)[::-1])
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.z_prg = Normal(self.prg, device=self.device)

    def modify_randomness(self, new_key=None, new_seed=None):
        if new_key is not None:
            self.prg_key = new_key
        if new_seed is not None:
            self.prg_seed = new_seed

        self._reload_randomness()

    def _model_iteration(self, sample, t, variance_noise):
        # 1. predict noise residual
        with torch.no_grad():
            residual = self.model(sample, t).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        if isinstance(self.scheduler, DDIMScheduler):
            sample = self.scheduler.step(
                residual, t, sample, eta=1, variance_noise=variance_noise
            ).prev_sample
        elif isinstance(self.scheduler, DDPMScheduler):
            sample = self.scheduler.step(
                residual, t, sample, variance_noise=variance_noise
            ).prev_sample
        else:
            raise NotImplementedError(self.scheduler.__class__)
        return sample

    def _encoding_step(
        self,
        samples,
        i,
        t,
        variance_noise,
        variance_noise_1,
        sample,
        next_sample,
        encoded_message=None,
        to_hide=None,
    ):
        samples[i] = {}
        samples[i]["all0"] = next_sample

        # Our understanding is that we can only encode into one of the channels of the image
        idx_channel = 0

        # Create the "all-1s" variance noise and compute the next sample for it
        all1_variance_noise = variance_noise.clone().detach()
        all1_variance_noise[0][idx_channel] = variance_noise_1[0][idx_channel]
        samples[i]["all1"] = self._model_iteration(
            sample, t, all1_variance_noise)

        if to_hide is not None:
            # Perform the encoding on the variance noise and compute the next sample for it
            encoded_message = torch.tensor(
                list(encoded_message), device=self.device, dtype=torch.int
            ).reshape((256, 256))
            encoding_variance_noise = variance_noise.clone().detach()
            encoding_variance_noise[0][idx_channel] = torch.where(
                encoded_message == 1,
                variance_noise_1[0][idx_channel],
                encoding_variance_noise[0][idx_channel],
            )
            samples[i]["hidden"] = self._model_iteration(
                sample, t, encoding_variance_noise
            )

        return samples

    def _last_step(self, samples, i, t, variance_noise, next_sample, to_hide=None):
        # Record the samples at this step as well
        samples[i] = {}
        samples[i]["all0"] = next_sample

        # Compute the encoded sample from step t-2 at step t-1
        if to_hide is not None:
            samples[i]["hidden"] = self._model_iteration(
                samples[i - 1]["hidden"], t, variance_noise
            )

        # Compute the all-1s sample from step t-2 at step t-1
        samples[i]["all1"] = self._model_iteration(
            samples[i - 1]["all1"], t, variance_noise
        )

        return samples

    def generate(
        self,
        to_hide: Union[bytes, None] = None,
        use_ecc: bool = True,
        regenerate: bool = False,
    ) -> dict:
        results = {}

        # Prepare the data to hide if it exists:
        if to_hide is not None:
            if use_ecc:
                encoded_message = self.ecc.encode(to_hide, 256 * 256)
            else:
                encoded_message = bitarray()
                encoded_message.frombytes(to_hide)
        else:
            encoded_message = None

        self._reload_randomness()

        # Initial noise distribution
        sample = torch.randn(
            1,
            self.model.config.in_channels,
            self.model.config.sample_size,
            self.model.config.sample_size,
        )
        sample = sample.to(self.device)

        # Dict to hold information about the samples for encoding
        samples = {}

        # Generate the new samples for each iteration
        for i, t in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
            if not regenerate:
                # A production implementation would use the PRG for _every_ iteration
                # Our PRG code isn't very optimized, so we just use it for encoding
                # TODO: Optimize the PRG and replace it here
                if i == self.scheduler.num_inference_steps - 2:
                    self._time_benchmark("gen_variance_noise")
                    # Use PRG-generated randomness
                    variance_noise = self.z_prg.generate(
                        sample.shape, dtype=sample.dtype)
                    self._time_benchmark("gen_variance_noise")
                else:
                    # Use normal randomness
                    variance_noise = torch.randn(
                        sample.shape, dtype=sample.dtype)
                    variance_noise = variance_noise.to(self.device)

                next_sample = self._model_iteration(sample, t, variance_noise)
            elif i < self.scheduler.num_inference_steps - 2:
                # We have state, so we can regenerate from it
                continue

            # Encoding occurs at time (timesteps - 2)
            if i == self.scheduler.num_inference_steps - 2:
                # We now need to generate the sample for the actual encoding of an image
                if not regenerate:
                    self._time_benchmark("gen_variance_noise_1")
                    variance_noise_1 = self.z_prg.generate(
                        sample.shape, dtype=sample.dtype)
                    self._time_benchmark("gen_variance_noise_1")

                    # Save the current model state
                    self.generate_state[i] = {}
                    self.generate_state[i]["sample"] = sample
                    self.generate_state[i]["next_sample"] = next_sample
                    self.generate_state[i]["variance_noise"] = variance_noise
                    self.generate_state[i]["variance_noise_1"] = variance_noise_1
                else:
                    # Retrieve previous model state
                    variance_noise = self.generate_state[i]["variance_noise"]
                    variance_noise_1 = self.generate_state[i]["variance_noise_1"]
                    next_sample = self.generate_state[i]["next_sample"]
                    sample = self.generate_state[i]["sample"]

                samples = self._encoding_step(
                    samples,
                    i,
                    t,
                    variance_noise,
                    variance_noise_1,
                    sample,
                    next_sample,
                    encoded_message,
                    to_hide,
                )

            # We also need information at the last timestep
            elif i == self.scheduler.num_inference_steps - 1:
                if not regenerate:
                    # Save the current model state at this step
                    self.generate_state[i] = {}
                    self.generate_state[i]["variance_noise"] = variance_noise
                else:
                    # Retrieve previous model state at this step
                    variance_noise = self.generate_state[i]["variance_noise"]

                    # We still need to call the model here, since we don't do it above
                    next_sample = self._model_iteration(
                        sample, t, variance_noise)

                samples = self._last_step(
                    samples, i, t, variance_noise, next_sample, to_hide
                )

            sample = next_sample

        # Add the newly generated samples to the results
        results["samples"] = samples
        results["encoded_message"] = encoded_message

        return results

    def _collect_region_info(region_avg_errors, region_sizes, start_idx, end_idx):
        weighted_avg = (
            region_avg_errors[start_idx:end_idx]
            * region_sizes[start_idx:end_idx]
            / region_sizes[start_idx:end_idx].sum()
        )

        # Handle the case where the region was empty and we divided by zero
        collected_error_rate = torch.sum(
            torch.where(torch.isnan(weighted_avg), 0, weighted_avg)
        )
        collected_bit_size = region_sizes[start_idx:end_idx].sum()

        return (collected_error_rate, collected_bit_size)

    def estimate_regions(self, n_to_gen=1, n_hist_bins=100, end_to_end=False):
        # Benchmark information for the whole function
        self._time_benchmark("estimate_regions")

        all_generate_results = []
        all_reveal_results = []
        last = self.scheduler.num_inference_steps - 1

        # Make sure randomness state is current
        self._reload_randomness()

        # Benchmark information for the model iterations
        self._time_benchmark("model_iterations")

        for j in range(n_to_gen):
            # reuse state if it is available
            regenerate = last in self.generate_state

            if not regenerate:
                self._time_benchmark("initial_model_iteration")

            # random_message = os.urandom(256 * 256 // 8)
            random_message = np.packbits(
                self.prg.get_bits(256 * 256)).tobytes()

            generate_results = self.generate(
                random_message,
                regenerate=regenerate,
                use_ecc=False,
            )

            if not regenerate:
                self._time_benchmark("initial_model_iteration")

            all_generate_results.append(generate_results)

            hidden_sample = generate_results["samples"][last]["hidden"]
            all0_sample = generate_results["samples"][last]["all0"]
            all1_sample = generate_results["samples"][last]["all1"]
            encoded_message = generate_results["encoded_message"]

            if end_to_end:
                tmp_file = "/tmp/pulsar_estimate.png"
                self.save_sample(hidden_sample, tmp_file)
                hidden_sample = self.load_sample(tmp_file)
                # self.save_sample(all0_sample.reshape(
                #     (1, 3, 256, 256)), tmp_file)
                # all0_sample = self.load_sample(tmp_file)
                # self.save_sample(all1_sample.reshape(
                #     (1, 3, 256, 256)), tmp_file)
                # all1_sample = self.load_sample(tmp_file)

            reveal_results = self.reveal(
                hidden_sample,
                all0_sample,
                all1_sample,
                encoded_message=encoded_message,
                use_ecc=False,
            )
            all_reveal_results.append(reveal_results)

        # Benchmark information for the model iterations
        self._time_benchmark("model_iterations")

        # Compute the histogram
        all0_sample = reveal_results["pulsar"]["samples"]["all0"]
        all1_sample = reveal_results["pulsar"]["samples"]["all1"]

        if end_to_end:
            tmp_file = "/tmp/pulsar_estimate.png"
            self.save_sample(all0_sample.reshape((1, 3, 256, 256)), tmp_file)
            all0_sample = self.load_sample(
                tmp_file)[0]  # Need shape (3, 256, 256)
            self.save_sample(all1_sample.reshape((1, 3, 256, 256)), tmp_file)
            all1_sample = self.load_sample(
                tmp_file)[0]  # Need shape (3, 256, 256)

        all1_diff = abs(all1_sample[0] - all0_sample[0]).flatten().cpu()
        (region_sizes, region_bins) = torch.histogram(
            all1_diff, bins=n_hist_bins)

        # Assign each pixel to a difference region by associating it to a bin from the histogram
        pixel_assignments = (
            torch.bucketize(all1_diff, region_bins, right=True) - 1
        )  # 1 indexed
        pixel_assignments = pixel_assignments.reshape(
            (256, 256)).to(self.device)

        # Determine for each generated image the error rates in the predicted regions
        # Need to make sure the device is the same (the MPS backend fails silently otherwise)
        region_errors = torch.zeros(
            (n_to_gen, n_hist_bins), device=self.device)

        for i in range(n_to_gen):
            reveal_results = all_reveal_results[i]
            hidden_sample = reveal_results["pulsar"]["samples"]["hidden"]

            errors_matrix = reveal_results["pulsar"]["errors"]

            # We need to calculate the error rate for the estimated pixel assignments
            for j in range(0, n_hist_bins):
                region_errors[i][j] = (
                    torch.where(pixel_assignments == j, errors_matrix, 0).sum()
                    / region_sizes[j]
                )

        # Take the average error rate for the estimated regions across all of the generated images
        region_avg_errors = region_errors.sum(axis=0) / n_to_gen

        # Make sure everything is on the same device
        region_sizes = region_sizes.to(self.device)

        self._calc_regions(region_avg_errors, region_sizes, pixel_assignments)

        self.all0_sample = all0_sample
        self.all1_sample = all1_sample

        # Maximum message len = sum of message len of regions
        # Message len of regions = number of sage calls per region * input size of region

        self.max_message_len = sum(
            [
                (x["size"] // x["code"]["output_size"]) * x["code"]["input_size"]
                for x in self.regions
            ]
        )

        # Benchmark information for the whole function
        self._time_benchmark("estimate_regions")

        # Benchmark information for the message length
        self._multi_benchmark("max_message_len", self.max_message_len)

        return self.max_message_len

    def _calc_regions(self, region_avg_errors, region_sizes, pixel_assignments):
        # The pixel to bin assignments will be useful for the encoding step
        pixel_assignments = pixel_assignments.cpu()

        # Start from the last bin indices
        # Because of how bucketize works, the last bin has index n_hist_bins + 1
        bin_end_idx = len(region_sizes) + 1
        bin_start_idx = bin_end_idx - 1
        regions = []

        while bin_start_idx >= 0:
            # Collect the bits in the bins so far
            collected_error_rate, collected_bit_size = Pulsar._collect_region_info(
                region_avg_errors, region_sizes, bin_start_idx, bin_end_idx
            )

            # Check if the collected error rate and bit size are compatible with a code from the library
            # We assume that the smallest error rate codes have the best bit rates, so we take a greedy approach here
            # TODO: Optimize
            code_lib_error_rates = sorted(SageCode.CODE_LIBRARY.keys())
            for code_error_rate in code_lib_error_rates:
                if collected_error_rate > code_error_rate:
                    continue

                code_params = SageCode.CODE_LIBRARY[code_error_rate][0]

                if collected_bit_size < code_params["output_size"]:
                    continue

                region_mask = (pixel_assignments >= bin_start_idx) & (
                    pixel_assignments < bin_end_idx
                )

                # The collected bins are under the error rate of a library code, and there are enough bits in the bin to encode
                regions.append(
                    {
                        "start": bin_start_idx,
                        "end": bin_end_idx,
                        "error_rate": collected_error_rate.cpu().item(),
                        "size": collected_bit_size.int().cpu().item(),
                        "code": code_params,
                        "mask": region_mask,
                    }
                )

                # Start the search again with the remaining bins
                bin_end_idx = bin_start_idx

                break

            # Continue the search
            bin_start_idx -= 1

        self.regions = regions

    def generate_with_regions(self, message):
        self._time_benchmark("generate_with_regions")

        # Use Sage to encode a message based on the error rates
        encoded_message = np.zeros((256, 256), dtype=int)

        # Make sure randomness state is current
        self._reload_randomness()

        to_sage = []
        l = 0
        for region in self.regions:
            sage_input_size_bytes = region["code"]["input_size"]
            sage_code_size_bits = region["code"]["output_size"]
            region_size = region["size"]

            # Number of Sage encodings we can fit in a single image
            sage_calls = region_size // sage_code_size_bits
            capacity = sage_calls * sage_input_size_bytes
            output_size = sage_calls * sage_code_size_bits

            to_encode = bytearray(message[l: l + capacity])
            l += capacity

            # Pad out the message
            # TODO: We can possibly do better by re-using the "unused" portions of this region in the next region
            if len(to_encode) < capacity:
                to_encode.extend(
                    bytearray(b"\x00" * (capacity - len(to_encode))))

            to_encode = list(bytes(to_encode))
            assert len(to_encode) == capacity

            for i in range(0, capacity, sage_input_size_bytes):
                to_sage.append(
                    SageCode._make_code_info(
                        region["code"], to_encode, i, sage_input_size_bytes
                    )
                )

        self._time_benchmark("sage_encode")
        results = SageCode.call_sage("encode", to_sage)
        self._time_benchmark("sage_encode")
        result_idx = 0

        for region in self.regions:
            sage_code_size_bits = region["code"]["output_size"]
            region_size = region["size"]
            sage_calls = region_size // sage_code_size_bits
            output_size = sage_calls * sage_code_size_bits
            padding = region_size - output_size if region_size > output_size else 0

            result = bitarray()
            for i in range(sage_calls):
                result.extend(results[result_idx + i])
            result.extend(zeros(padding))

            # randomness state is in sync between sender and receiver
            p = np.random.permutation(len(result))
            # p, _ = Permutation(self.prg).get(len(result))
            result = permuted(result, p)

            region["encoded_message"] = np.array(list(result))

            np.place(encoded_message,
                     region["mask"], region["encoded_message"])

            result_idx += sage_calls

        # TODO: remove the need to do this conversion
        to_encode = bitarray(encoded_message.flatten().tolist()).tobytes()

        self._time_benchmark("model_embed")
        to_return = self.generate(to_encode, regenerate=True, use_ecc=False)
        self._time_benchmark("model_embed")
        self._time_benchmark("generate_with_regions")
        return to_return

    def _calc_distance(self, a, b, dims=3):
        if self.distance == "euclidean":
            # TODO: Fix to use torch
            return np.sqrt(sum([np.square(a[i] - b[i]) for i in range(0, dims)]))
        elif self.distance == "abs":
            return torch.abs(a - b)[0]

    def reveal(
        self,
        hidden_sample: torch.Tensor,
        all0_sample: torch.Tensor,
        all1_sample: torch.Tensor,
        encoded_message=None,
        use_ecc=True,
        return_bitarray=True,
        return_meta=True,
    ):
        results = {}

        hidden_sample = hidden_sample[0]
        all0_sample = all0_sample[0]
        all1_sample = all1_sample[0]

        distances = [
            self._calc_distance(hidden_sample, all0_sample),
            self._calc_distance(hidden_sample, all1_sample),
        ]

        # Image with the lower distance -> recovered bit
        # 0 if closer to all-0s image (distances[0] < distances[1])
        # 1 if closer to all-1s image (distances[0] >= distances[1])
        recovered = distances[0] >= distances[1]

        z = recovered.reshape((256 * 256,)).cpu().numpy()

        if return_bitarray:
            # Have to do this crazy string conversion to efficiently move from Numpy to Bitarray
            # TODO: Is there a better way?
            recovered_message = bitarray(
                np.array2string(
                    z,
                    prefix="",
                    suffix="",
                    separator="",
                    max_line_width=256 * 256 * 256,  # avoid newlines
                    threshold=256 * 256 * 256,  # avoid truncation
                    formatter={"bool": lambda x: "1" if x else "0"},
                )[1:-1]
            )
        else:
            recovered_message = z

        results["pulsar"] = {}
        results["pulsar"]["recovered"] = recovered_message

        if not return_meta:
            return results

        for i in range(len(recovered_message)):
            if z[i] and recovered_message[i] != 1:
                raise ValueError(i)

        if encoded_message is not None:
            # We'll use the encoded message to test for errors
            encoded_message = torch.tensor(
                list(encoded_message), device=self.device, dtype=torch.int
            ).reshape((256, 256))
            errors_matrix = encoded_message != recovered
            results["pulsar"]["errors"] = errors_matrix

        results["pulsar"]["samples"] = {}
        results["pulsar"]["samples"]["hidden"] = hidden_sample
        results["pulsar"]["samples"]["all0"] = all0_sample
        results["pulsar"]["samples"]["all1"] = all1_sample

        decoded_message, decoding_meta = (
            self.ecc.decode(recovered_message)
            if use_ecc
            else (recovered_message.tobytes(), None)
        )

        results["ecc"] = {}
        results["ecc"]["decoded"] = decoded_message
        results["ecc"]["meta"] = decoding_meta

        return results

    def reveal_with_regions(self, hidden_sample, encoded_message=None):
        self._time_benchmark("reveal_with_regions")

        self._time_benchmark("model_de_embed")
        reveal_results = self.reveal(
            hidden_sample,
            self.all0_sample,
            self.all1_sample,
            encoded_message,
            use_ecc=False,
            return_bitarray=False,
            return_meta=False,
        )
        self._time_benchmark("model_de_embed")

        # Extract the message components from the tensor and decode
        # return_bitarray=False above means recovered_message is a Numpy array
        recovered_message = reveal_results["pulsar"]["recovered"]

        message = bytearray()
        to_sage = []

        # Make sure random number state is in sync
        self._reload_randomness()

        for region in self.regions:
            sage_code_size_bits = region["code"]["output_size"]
            region_size = region["size"]
            sage_calls = region_size // sage_code_size_bits
            output_size = sage_calls * sage_code_size_bits

            to_decode = np.extract(region["mask"], recovered_message)

            # Invert the random permutation
            p = np.random.permutation(len(to_decode))
            p_inv = np.argsort(p)
            # _, p_inv = Permutation(self.prg).get(len(to_decode))
            to_decode = permuted(to_decode, p_inv)

            # Remove padding from the region
            to_decode = to_decode[:output_size]
            to_decode = list(to_decode)

            # The region is made up of one or more calls to Sage
            for i in range(0, output_size, sage_code_size_bits):
                to_sage.append(
                    SageCode._make_code_info(
                        region["code"], to_decode, i, sage_code_size_bits
                    )
                )

        self._time_benchmark("sage_decode")
        from_sage = SageCode.call_sage("decode", to_sage)
        self._time_benchmark("sage_decode")

        for decoded in from_sage:
            message.extend(decoded)

        self._time_benchmark("reveal_with_regions")
        return bytes(message)


# Monkey-patch the DDPM scheduler to make it easier to modify the variance noise
# TODO: convert this to an approach more in line with the diffusers library
def _ddpm_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    variance_noise,  # this was a torch.Generator in the original
    return_dict: bool = True,
) -> Union[DDPMSchedulerOutput, Tuple]:
    """
    A modified DDPMScheduler.step() that accepts variance noise as a parameter rather than a torch.Generator
    This does not change anything fundamental about DDPM, just the interface to the variance noise
    Original https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/schedulers/scheduling_ddpm.py#L312
    TODO: Create a torch.Generator so we don't have to monkey-patch
    """

    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * \
        beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = variance_noise.to(device)
        if self.variance_type == "fixed_small_log":
            variance = (
                self._get_variance(t, predicted_variance=predicted_variance)
                * variance_noise
            )
        elif self.variance_type == "learned_range":
            variance = self._get_variance(
                t, predicted_variance=predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (
                self._get_variance(
                    t, predicted_variance=predicted_variance) ** 0.5
            ) * variance_noise

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample,)

    return DDPMSchedulerOutput(
        prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
    )


DDPMScheduler.step = _ddpm_step
