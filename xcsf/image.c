/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file image.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Image handling functions.
 */

static void
col2im_add_pixel(double *im, const int height, const int width, int row,
                 int col, const int channel, const int pad, const double val)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) {
        return;
    }
    im[col + width * (row + height * channel)] += val;
}

static double
im2col_get_pixel(const double *im, const int height, const int width, int row,
                 int col, const int channel, const int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) {
        return 0;
    }
    return im[col + width * (row + height * channel)];
}

/**
 * @brief Transforms a column matrix to an image matrix.
 * @details Used for GEMM convolutional backward propagation.
 * @param data_col Input column matrix.
 * @param channels Number of image channels.
 * @param height Image height.
 * @param width Image width.
 * @param ksize Kernel size.
 * @param stride Kernel stride.
 * @param pad Kernel padding.
 * @param data_im The resulting image matrix.
 */
void
col2im(const double *data_col, const int channels, const int height,
       const int width, const int ksize, const int stride, const int pad,
       double *data_im)
{
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        const int w_offset = c % ksize;
        const int h_offset = (c / ksize) % ksize;
        const int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int im_row = h_offset + h * stride;
                const int im_col = w_offset + w * stride;
                const int col_index = (c * height_col + h) * width_col + w;
                const double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, im_row, im_col, c_im,
                                 pad, val);
            }
        }
    }
}

/**
 * @brief Transforms an image matrix to a column matrix.
 * @details Used for GEMM convolutional forward propagation.
 * @param data_im Image matrix of dimension: height × width × channels.
 * @param channels Number of image channels.
 * @param height Image height.
 * @param width Image width.
 * @param ksize Kernel size.
 * @param stride Kernel stride.
 * @param pad Kernel padding.
 * @param data_col The resulting column matrix.
 */
void
im2col(const double *data_im, const int channels, const int height,
       const int width, const int ksize, const int stride, const int pad,
       double *data_col)
{
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        const int w_offset = c % ksize;
        const int h_offset = (c / ksize) % ksize;
        const int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int im_row = h_offset + h * stride;
                const int im_col = w_offset + w * stride;
                const int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(
                    data_im, height, width, im_row, im_col, c_im, pad);
            }
        }
    }
}
