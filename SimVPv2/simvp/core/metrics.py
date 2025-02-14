import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
from scipy.linalg import sqrtm


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

# calculate frechet inception distance
def FID(pred, true):
    fids = np.zeros(true.shape[0])
    true = true[:,0,0,:]
    pred = pred[:,0,0,:]
    for i in range(true.shape[0]):
        # Convert grayscale images to RGB by duplicating the channels
        if true[i].shape[1] == 1:
            true[i] = np.repeat(true[i], 3, axis=1)
        if pred[i].shape[1] == 1:
            pred[i] = np.repeat(pred[i], 3, axis=1)

        # Calculate the mean and covariance of true images
        true_mean = np.mean(true[i], axis=0)
        true_cov = np.cov(true[i], rowvar=False)

        # Calculate the mean and covariance of generated images
        generated_mean = np.mean(pred[i], axis=0)
        generated_cov = np.cov(pred[i], rowvar=False)

        # Calculate the squared distance between means
        mean_diff = true_mean - generated_mean
        mean_diff_squared = np.dot(mean_diff, mean_diff)

        # Calculate the trace of the product of covariances
        cov_product = np.dot(true_cov, generated_cov)
        # Note that this can fail for some matrices. If it does, we will return nan as the fid value.
        # https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/linalg/_matfuncs_sqrtm.py#L194
        cov_product_sqrt = sqrtm(cov_product)
        trace_cov_product_sqrt = np.real(np.trace(cov_product_sqrt))

        # Calculate the FID score
        fid = mean_diff_squared + np.trace(true_cov) + np.trace(generated_cov) - 2 * trace_cov_product_sqrt
        fids[i] = fid

    return np.mean(fids)


# Calculate the MSE of the prediction against the last input frame
# Use case: If the prev_frame_mse > mse, then it may indicate that the prediction is not simply trying to reconstruct the last input frame
def prev_frame_MSE(pred, inputs):
    inputs = inputs[:,-1,:,:]
    pred = pred[:,-1,:,:]
    return np.mean((pred-inputs)**2, axis=(0, 1)).sum()

def metric(pred, true, mean, std, inputs=None, metrics=['mae', 'mse'],
        clip_range=[0, 1], spatial_norm=False):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        inputs (tensor): The input values for the prediction
    Returns:
        dict: evaluation results
    """
    pred = pred * std + mean
    true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'fid', 'prev_frame_mse']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, true, spatial_norm)

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, true, spatial_norm)

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, true, spatial_norm)

    if 'fid' in metrics:
        eval_res['fid'] = FID(pred, true)

    if 'prev_frame_mse' in metrics and inputs is not None:
        eval_res['prev_frame_mse'] = prev_frame_MSE(pred, inputs)

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
       ssim = 0
       for b in range(pred.shape[0]):
           for f in range(pred.shape[1]):
               ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                true[b, f].swapaxes(0, 2), channel_axis=2, data_range=1)
       eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    for k, v in eval_res.items():
        eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
        eval_log += eval_str

    return eval_res, eval_log
