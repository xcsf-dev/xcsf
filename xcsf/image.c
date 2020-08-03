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
col2im_add_pixel(double *im, int height, int width, int row, int col,
                 int channel, int pad, double val)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) {
        return;
    }
    im[col + width * (row + height * channel)] += val;
}

static double
im2col_get_pixel(const double *im, int height, int width, int row, int col,
                 int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) {
        return 0;
    }
    return im[col + width * (row + height * channel)];
}

void
col2im(const double *data_col, int channels, int height, int width, int ksize,
       int stride, int pad, double *data_im)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, im_row, im_col, c_im,
                                 pad, val);
            }
        }
    }
}

void
im2col(const double *data_im, int channels, int height, int width, int ksize,
       int stride, int pad, double *data_col)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(
                    data_im, height, width, im_row, im_col, c_im, pad);
            }
        }
    }
}
