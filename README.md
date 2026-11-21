# Recurrent neural networks: Hopfield and Elman on handwritten digits

The project contains a Hopfield network (associative memory), an Elman network (simple recurrent network) and a
multilayer perceptron for comparison. Everything
is implemented from scratch with NumPy.

Dataset: the 8x8 handwritten digits shipped with scikit-learn (1797 images,
10 classes, pixel values 0..16). The data is split 70/30 (stratified, seed 42)
into 1257 training and 540 test images. The same task is solved by all networks:

* Hopfield network: 64 neurons (one per pixel). One bipolar prototype (+1 / -1)
  per digit is built from the training images, the weights are computed in one
  step (Hebb or pseudoinverse rule). An image is recognized by recalling the
  nearest prototype. Noise can be added to the inputs.
* Elman network: the image is read row by row (8 steps with 8 pixels each),
  16 hidden tanh neurons with a context layer, 10 tanh outputs. Trained by
  backpropagation through time with gradient clipping.
* Multilayer perceptron (baseline): the whole image at once (64 inputs),
  16 hidden tanh neurons, 10 tanh outputs.

The Elman network and the perceptron use bipolar one-hot targets and the same
optimizers (SGD, Adam), so they can be compared fairly.

## Setup

Python 3.12. From the project root:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

The digits are loaded from scikit-learn, no files need to be downloaded.

## Project structure

    recurrent_lab/          the library
        config.py           all settings in one place
        data.py             loading, split, bipolar encoding, sequences, noise
        elman.py            Elman network, backpropagation through time
        evaluation.py       error and classification time of a model
        hopfield.py         Hopfield network (Hebb and pseudoinverse rules)
        mlp.py              multilayer perceptron with backpropagation
        optimizers.py       SGD, SGD with momentum, RMSprop, Adam
        runner.py           train, evaluate and save one experiment
        storage.py          saving a run to disk
        utils.py            timer, error, training result
    build_summary.py        summary table (CSV and Excel)
    check_gradients.py      numerical check of the backpropagation gradients
    plot_results.py         all graphs of the report
    run_elman.py            experiment group B (optimizers, hidden sizes)
    run_hopfield.py         experiment group A (learning rules, noise study)
    run_mlp_baseline.py     experiment group C (feedforward baseline)

## How to reproduce the results

Run the scripts from the project root in this order:

    python check_gradients.py       # optional: gradients must match numerically
    python run_hopfield.py          # a few seconds
    python run_elman.py             # about 2 minutes
    python run_mlp_baseline.py      # about 1 minute
    python build_summary.py         # needs the three groups above
    python plot_results.py          # redraws all graphs in seconds

The graphs are in Ukrainian by default. Set `LANGUAGE = "en"` in
`plot_results.py` for English.

## Results

Everything is written to `results/`:

    results/runs/<run_id>/      one folder per run: config.json, metrics.json,
                                initial_params.npz (not for the Hopfield
                                network), trained_params.npz,
                                predictions_train.csv, predictions_test.csv
    results/hopfield_noise_study.csv   Hopfield error against the noise level
    results/summary.csv         all runs, English column names
    results/summary.xlsx        the report table, Ukrainian column names
    results/figures/            graphs

Test error with the default settings (seed 42):

    Hopfield, Hebb rule              90.4 %
    Hopfield, pseudoinverse rule     23.9 %
    Elman network, 16 hidden         5.2 - 5.7 %
    Multilayer perceptron, 16 hidden 2.4 - 3.2 %

## Notes

* The Hopfield network has no epochs: its weights are computed in one step.
  The Hebb rule fails on the digits: the prototypes are strongly correlated
  and their number (10) is above the capacity of 64 neurons (about 0.14 * N),
  so every input settles into the same wrong state. The pseudoinverse rule
  stores all 10 prototypes exactly.
* The Hopfield error on real images is high because every class is represented
  by a single prototype, and handwritten digits differ from it.
* The training time of the Elman network and the perceptron is the time until
  the stopping criterion fires (the loss has stopped improving), not the time
  to the exact minimum.
* The results are reproducible with the same seeds; the training time differs
  a little from run to run, so it is averaged over repeats.