import numpy as np
import matplotlib.pyplot as plt

class ImagePatcher:
    def __init__(self, patch_size, overlap):
        self.patch_size = patch_size
        self.overlap = overlap
        self.original_shape = None

    def split(self, image):
        self.original_shape = image.shape
        height, width, channels = image.shape

        # Calculate dynamic stride
        stride_y = self.patch_size - self.overlap
        stride_x = self.patch_size - self.overlap

        patches = []
        for i in range(0, height - self.patch_size + 1, stride_y):
            for j in range(0, width - self.patch_size + 1, stride_x):
                patch = image[i:i + self.patch_size, j:j + self.patch_size, :]
                patches.append(patch)

        # Handle edge cases
        if height % stride_y != 0:
            for j in range(0, width - self.patch_size + 1, stride_x):
                patches.append(image[-self.patch_size:, j:j + self.patch_size, :])

        if width % stride_x != 0:
            for i in range(0, height - self.patch_size + 1, stride_y):
                patches.append(image[i:i + self.patch_size, -self.patch_size:, :])

        if height % stride_y != 0 and width % stride_x != 0:
            patches.append(image[-self.patch_size:, -self.patch_size:, :])

        return np.array(patches)

    def merge(self, patches):
        if self.original_shape is None:
            raise ValueError("No original image information available. Call split() first.")

        height, width, channels = self.original_shape
        reconstructed = np.zeros((height, width, channels))
        count = np.zeros((height, width, channels))

        stride_y = self.patch_size - self.overlap
        stride_x = self.patch_size - self.overlap

        idx = 0
        for i in range(0, height - self.patch_size + 1, stride_y):
            for j in range(0, width - self.patch_size + 1, stride_x):
                reconstructed[i:i + self.patch_size, j:j + self.patch_size, :] += patches[idx]
                count[i:i + self.patch_size, j:j + self.patch_size, :] += 1
                idx += 1

        # Handle edge cases
        if height % stride_y != 0:
            for j in range(0, width - self.patch_size + 1, stride_x):
                reconstructed[-self.patch_size:, j:j + self.patch_size, :] += patches[idx]
                count[-self.patch_size:, j:j + self.patch_size, :] += 1
                idx += 1

        if width % stride_x != 0:
            for i in range(0, height - self.patch_size + 1, stride_y):
                reconstructed[i:i + self.patch_size, -self.patch_size:, :] += patches[idx]
                count[i:i + self.patch_size, -self.patch_size:, :] += 1
                idx += 1

        if height % stride_y != 0 and width % stride_x != 0:
            reconstructed[-self.patch_size:, -self.patch_size:, :] += patches[idx]
            count[-self.patch_size:, -self.patch_size:, :] += 1

        # Normalize the overlapping regions
        reconstructed = np.divide(reconstructed, count, where=count != 0)

        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def plot_patches(self, patches, max_patches=25):
        n = min(len(patches), max_patches)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        plt.figure(figsize=(15, 15))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            patch = patches[i]
            if patch.shape[2] > 3:
                plt.imshow(patch[:, :, :3].astype(np.uint8))
                plt.title(f'Patch {i+1}\n(first 3 channels)')
            elif patch.shape[2] == 3:
                plt.imshow(patch.astype(np.uint8))
                plt.title(f'Patch {i+1}\n(all 3 channels)')
            else:
                plt.imshow(patch[:, :, 0], cmap='gray')
                plt.title(f'Patch {i+1}\n(single channel)')
            plt.axis('off')

        plt.tight_layout()
        plt.show()