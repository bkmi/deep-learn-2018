import numpy as np


def load(move_channels=False):
    def swap(data):
        if move_channels:
            data = np.moveaxis(data, 1, -1)
        else:
            pass
        return data

    with np.load('denoising-challenge-01-data.npz') as fh:
        training_images_clean = swap(fh['training_images_clean'])
        validation_images_noisy = swap(fh['validation_images_noisy'])
        validation_images_clean = swap(fh['validation_images_clean'])
        test_images_noisy = swap(fh['test_images_noisy'])

    # TRAINING DATA: CLEAN
    # 1. INDEX: IMAGE SERIAL NUMBER (20000)
    # 2. INDEX: COLOR CHANNEL (1)
    # 3/4. INDEX: PIXEL VALUE (28 x 28)
    print('Training Clean', training_images_clean.shape, training_images_clean.dtype)

    # VALIDATION DATA: CLEAN + NOISY
    print('Validation Clean', validation_images_clean.shape, validation_images_clean.dtype)
    print('Validation Noise', validation_images_noisy.shape, validation_images_noisy.dtype)

    # TEST DATA: NOISY
    print('Test Noisy', test_images_noisy.shape, test_images_noisy.dtype)

    return training_images_clean, validation_images_noisy, validation_images_clean, test_images_noisy

# TRAIN MODEL ON training_images_clean

# CHECK YOUR MODEL USING (validation_images_clean, validation_images_noisy)

# DENOISE IMAGES (test_images_clean) USING test_images_noisy


def save(test_images_clean):
    # MAKE SURE THAT YOU HAVE THE RIGHT FORMAT
    assert test_images_clean.ndim == 4
    assert test_images_clean.shape[0] == 2000
    assert test_images_clean.shape[1] == 1
    assert test_images_clean.shape[2] == 28
    assert test_images_clean.shape[3] == 28

    # AND SAVE EXACTLY AS SHOWN BELOW
    np.save('test_images_clean.npy', test_images_clean)
