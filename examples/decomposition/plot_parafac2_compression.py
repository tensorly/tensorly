"""
Speeding up PARAFAC2 with SVD compression
=========================================

PARAFAC2 can be very time-consuming to fit. However, if the number of rows greatly
exceeds the number of columns or the data matrices are approximately low-rank, we can
compress the data before fitting the PARAFAC2 model to considerably speed up the fitting
procedure.

The compression works by first computing the SVD of the tensor slices and fitting the
PARAFAC2 model to the right singular vectors multiplied by the singular values. Then,
after we fit the model, we left-multiply the :math:`B_i`-matrices with the left singular
vectors to recover the decompressed model. Fitting to compressed data and then
decompressing is mathematically equivalent to fitting to the original uncompressed data.

For more information about why this works, see the documentation of
:py:meth:`tensorly.decomposition.preprocessing.svd_compress_tensor_slices`.
"""
from time import monotonic
import tensorly as tl
from tensorly.decomposition import parafac2
import tensorly.preprocessing as preprocessing


##############################################################################
# Function to create synthetic data
# ---------------------------------
#
# Here, we create a function that constructs a random tensor from a PARAFAC2
# decomposition with noise

rng = tl.check_random_state(0)


def create_random_data(shape, rank, noise_level):
    I, J, K = shape  # noqa: E741
    pf2 = tl.random.random_parafac2(
        [(J, K) for i in range(I)], rank=rank, random_state=rng
    )

    X = pf2.to_tensor()
    X_norm = [tl.norm(Xi) for Xi in X]

    noise = [rng.standard_normal((J, K)) for i in range(I)]
    noise = [noise_level * X_norm[i] / tl.norm(E_i) for i, E_i in enumerate(noise)]
    return [X_i + E_i for X_i, E_i in zip(X, noise)]


##############################################################################
# Compressing data with many rows and few columns
# -----------------------------------------------
#
# Here, we set up for a case where we have many rows compared to columns

n_inits = 5
rank = 3
shape = (10, 10_000, 15)  # 10 matrices/tensor slices, each of size 10_000 x 15.
noise_level = 0.33

uncompressed_data = create_random_data(shape, rank=rank, noise_level=noise_level)

##############################################################################
# Fitting without compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As a baseline, we see how long time it takes to fit models without compression.
# Since PARAFAC2 is very prone to local minima, we fit five models and select the model
# with the lowest reconstruction error.

print("Fitting PARAFAC2 model without compression...")
t1 = monotonic()
lowest_error = float("inf")
for i in range(n_inits):
    pf2, errs = parafac2(
        uncompressed_data,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_full, errs_full = pf2, errs
t2 = monotonic()
print(
    f"It took {t2 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "without compression"
)

##############################################################################
# Fitting with lossless compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Since the tensor slices have many rows compared to columns, we should be able to save
# a lot of time by compressing the data. By compressing the matrices, we only need to
# fit the PARAFAC2 model to a set of 10 matrices, each of size 15 x 15, not 10_000 x 15.
#
# The main bottleneck here is the SVD computation at the beginning of the fitting
# procedure, but luckily, this is independent of the initialisations, so we only need
# to compute this once. Also, if we are performing a grid search for the rank, then
# we just need to perform the compression once for the whole grid search as well.

print("Fitting PARAFAC2 model with SVD compression...")
t1 = monotonic()
lowest_error = float("inf")
scores, loadings = preprocessing.svd_compress_tensor_slices(uncompressed_data)
t2 = monotonic()
for i in range(n_inits):
    pf2, errs = parafac2(
        scores,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_compressed, errs_compressed = pf2, errs
pf2_decompressed = preprocessing.svd_decompress_parafac2_tensor(
    pf2_compressed, loadings
)
t3 = monotonic()
print(
    f"It took {t3 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "with lossless SVD compression"
)
print(f"The compression took {t2 - t1:.1f}s and the fitting took {t3 - t2:.1f}s")

##############################################################################
# We see that we saved a lot of time by compressing the data before fitting the model.

##############################################################################
# Fitting with lossy compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can try to speed the process up even further by accepting a slight discrepancy
# between the model obtained from compressed data and a model obtained from uncompressed
# data. Specifically, we can truncate the singular values at some threshold, essentially
# removing the parts of the data matrices that have a very low "signal strength".

print("Fitting PARAFAC2 model with lossy SVD compression...")
t1 = monotonic()
lowest_error = float("inf")
scores, loadings = preprocessing.svd_compress_tensor_slices(uncompressed_data, 1e-5)
t2 = monotonic()
for i in range(n_inits):
    pf2, errs = parafac2(
        scores,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_compressed_lossy, errs_compressed_lossy = pf2, errs
pf2_decompressed_lossy = preprocessing.svd_decompress_parafac2_tensor(
    pf2_compressed_lossy, loadings
)
t3 = monotonic()
print(
    f"It took {t3 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "with lossy SVD compression"
)
print(
    f"Of which the compression took {t2 - t1:.1f}s and the fitting took {t3 - t2:.1f}s"
)

##############################################################################
# We see that we didn't save much, if any, time in this case (compared to using
# lossless compression). This is because the main bottleneck now is the CP-part of
# the PARAFAC2 procedure, so reducing the tensor size from 10 x 15 x 15 to 10 x 4 x 15
# (which is typically what we would get here) will have a negligible effect.


##############################################################################
# Compressing data that is approximately low-rank
# -----------------------------------------------
#
# Here, we simulate data with many rows and columns but an approximately low rank.

rank = 3
shape = (10, 2_000, 2_000)
noise_level = 0.33

uncompressed_data = create_random_data(shape, rank=rank, noise_level=noise_level)

##############################################################################
# Fitting without compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Again, we start by fitting without compression as a baseline.

print("Fitting PARAFAC2 model without compression...")
t1 = monotonic()
lowest_error = float("inf")
for i in range(n_inits):
    pf2, errs = parafac2(
        uncompressed_data,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_full, errs_full = pf2, errs
t2 = monotonic()
print(
    f"It took {t2 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "without compression"
)

##############################################################################
# Fitting with lossless compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we fit with lossless compression.

print("Fitting PARAFAC2 model with SVD compression...")
t1 = monotonic()
lowest_error = float("inf")
scores, loadings = preprocessing.svd_compress_tensor_slices(uncompressed_data)
t2 = monotonic()
for i in range(n_inits):
    pf2, errs = parafac2(
        scores,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_compressed, errs_compressed = pf2, errs
pf2_decompressed = preprocessing.svd_decompress_parafac2_tensor(
    pf2_compressed, loadings
)
t3 = monotonic()
print(
    f"It took {t3 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "with lossless SVD compression"
)
print(
    f"Of which the compression took {t2 - t1:.1f}s and the fitting took {t3 - t2:.1f}s"
)

##############################################################################
# We see that the lossless compression no effect for this data. This is because the
# number ofrows is equal to the number of columns, so we cannot compress the data
# losslessly with the SVD.

##############################################################################
# Fitting with lossy compression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, we fit with lossy SVD compression.

print("Fitting PARAFAC2 model with lossy SVD compression...")
t1 = monotonic()
lowest_error = float("inf")
scores, loadings = preprocessing.svd_compress_tensor_slices(uncompressed_data, 1e-5)
t2 = monotonic()
for i in range(n_inits):
    pf2, errs = parafac2(
        scores,
        rank,
        n_iter_max=1000,
        nn_modes=[0],
        random_state=rng,
        return_errors=True,
    )
    if errs[-1] < lowest_error:
        pf2_compressed_lossy, errs_compressed_lossy = pf2, errs
pf2_decompressed_lossy = preprocessing.svd_decompress_parafac2_tensor(
    pf2_compressed_lossy, loadings
)
t3 = monotonic()
print(
    f"It took {t3 - t1:.1f}s to fit a PARAFAC2 model a tensor of shape {shape} "
    + "with lossy SVD compression"
)
print(
    f"Of which the compression took {t2 - t1:.1f}s and the fitting took {t3 - t2:.1f}s"
)


##############################################################################
# Here we see a large speedup. This is because the data is approximately low rank so
# the compressed tensor slices will have shape R x 2_000, where R is typically below 10
# in this example. If your tensor slices are large in both modes, you might want to plot
# the singular values of your dataset to see if lossy compression could speed up
# PARAFAC2.
