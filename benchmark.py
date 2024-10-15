import json
import time
import copy
import os

import torch
from diffusers import DDIMScheduler, DDPMScheduler

import pulsar

n_iter = 100

cuda_benchmarks = [
    # Find the number of histogram bins
    {
        "name": "n_hist_bins",
        "n_iter": n_iter,
        "models": ["ddpm-church-256"],
        "ns_to_gen": [1],
        "ns_hist_bins": [25, 50, 100, 125, 150],
        "end_to_end": False,
        "scheduler": "DDIMScheduler",
        "num_inference_steps": 50,
    },
    # Find the number of estimates to make
    {
        "name": "n_to_gen",
        "n_iter": n_iter,
        "models": ["ddpm-church-256"],
        "ns_to_gen": [1, 3, 5, 10, 30],
        "ns_hist_bins": [100],
        "end_to_end": False,
        "scheduler": "DDIMScheduler",
        "num_inference_steps": 50,
    },
    # Generate images for one model using the slower DDPM scheduler
    {
        "name": "ddpm",
        "n_iter": n_iter,
        "models": ["ddpm-church-256"],
        "ns_to_gen": [1],
        "ns_hist_bins": [100],
        "end_to_end": True,
        "scheduler": "DDPMScheduler",
        "num_inference_steps": 1000,
    },
    # Generate images for all models
    {
        "name": "models",
        "n_iter": n_iter,
        "models": [
            "ddpm-church-256",
            "ddpm-celebahq-256",
            "ddpm-bedroom-256",
            "ddpm-cat-256",
        ],
        "ns_to_gen": [1],
        "ns_hist_bins": [100],
        "end_to_end": True,
        "scheduler": "DDIMScheduler",
        "num_inference_steps": 50,
    },
]

# Only run the DDPM benchmark
# cuda_benchmarks = [cuda_benchmarks[-2]]

# Only run the image generation benchmark on mps
mps_benchmarks = [cuda_benchmarks[-1]]


def run_benchmark(
    name,
    n_iter,
    models,
    ns_to_gen,
    ns_hist_bins,
    end_to_end,
    scheduler,
    num_inference_steps,
):
    timestamp = int(time.time())
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    bench_results = {
        "params": {
            "n_iter": n_iter,
            "models": models,
            "ns_to_gen": ns_to_gen,
            "ns_hist_bins": ns_hist_bins,
            "end_to_end": end_to_end,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "device": device,
        }
    }

    # Configure scheduler to be used
    if scheduler == "DDIMScheduler":
        scheduler = DDIMScheduler
    elif scheduler == "DDPMScheduler":
        scheduler = DDPMScheduler
    else:
        raise NotImplementedError(scheduler)

    fname = "bench-results/pulsar_benchmark_{}_{}_{}.json".format(
        timestamp, name, device
    )

    for i in range(n_iter):
        seed = "{}".format(i).encode("utf-8")
        for model in models:
            sender_stego = pulsar.Pulsar(
                seed=seed,
                repo="google/" + model,
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
            )
            for n_to_gen in ns_to_gen:
                for n_hist_bins in ns_hist_bins:
                    params_str = "{}_{}_{}".format(model, n_to_gen, n_hist_bins)

                    if params_str not in bench_results:
                        bench_results[params_str] = []
                        if end_to_end:
                            bench_results[params_str + "_receiver"] = []

                    sender_stego.benchmarks = {}
                    print(
                        "benchmark",
                        name,
                        "iteration",
                        i + 1,
                        "of",
                        n_iter,
                        "params",
                        params_str,
                    )

                    try:
                        m_len = sender_stego.estimate_regions(
                            n_to_gen=n_to_gen,
                            n_hist_bins=n_hist_bins,
                            end_to_end=end_to_end,
                        )
                        message = os.urandom(m_len)
                        generate_results = sender_stego.generate_with_regions(message)
                    except Exception:
                        # Handle the case where Sage fails (SIGSEGV?) with ValueError
                        # or because the torch.histogram fails with a RuntimeError [nan, nan]
                        bench_results[params_str].append({"encoding_error": [1]})
                        sender_stego.benchmarks = {}
                        continue

                    last = sender_stego.scheduler.num_inference_steps - 1
                    hidden_sample = generate_results["samples"][last]["hidden"]

                    if end_to_end:
                        image_fname = "bench-results/{}/{}_{}_{}_{}_{:03}.png".format(
                            model, timestamp, name, n_to_gen, n_hist_bins, i
                        )
                        sender_stego.save_sample(hidden_sample, image_fname)

                        # We need to load the hidden sample so we can test decoding on the sender end
                        hidden_sample = sender_stego.load_sample(image_fname)

                    try:
                        decoded = sender_stego.reveal_with_regions(hidden_sample)
                        benchmarks = copy.deepcopy(sender_stego.benchmarks)
                        bench_results[params_str].append(benchmarks)
                    except ValueError:
                        bench_results[params_str].append({"decoding_error": [1]})
                    except AssertionError:
                        bench_results[params_str].append({"message_error": [1]})

                    if end_to_end:
                        # Record the filename saved
                        bench_results[params_str][-1]["fname"] = image_fname

                        # Recover the hidden sample from an image
                        receiver_stego = pulsar.Pulsar(
                            seed=seed,
                            repo="google/" + model,
                            scheduler=scheduler,
                            num_inference_steps=num_inference_steps,
                        )
                        receiver_stego.estimate_regions(
                            n_to_gen=n_to_gen,
                            n_hist_bins=n_hist_bins,
                            end_to_end=end_to_end,
                        )
                        received_hidden_sample = receiver_stego.load_sample(image_fname)

                        # Wrap reveal in a try block to catch errors
                        try:
                            decoded = receiver_stego.reveal_with_regions(
                                received_hidden_sample
                            )
                            assert decoded == message
                            # Add the receiver benchmarks to the latest entry
                            bench_results[params_str + "_receiver"].append(
                                receiver_stego.benchmarks
                            )
                        except ValueError:
                            bench_results[params_str + "_receiver"].append(
                                {"decoding_error": [1]}
                            )
                        except AssertionError:
                            bench_results[params_str + "_receiver"].append(
                                {"message_error": [1]}
                            )

        # Save intermediate results
        with open(fname, "w") as f:
            json.dump(bench_results, f)

    return fname


if __name__ == "__main__":
    if torch.cuda.is_available():
        for b in cuda_benchmarks[:]:
            run_benchmark(**b)
    else:
        # Run mps_benchmarks if cuda is unavailable
        for b in mps_benchmarks[:]:
            run_benchmark(**b)
