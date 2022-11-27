import taichi as ti
import numpy as np

ti.init(default_fp=ti.f32)

@ti.kernel
def sobel_x_filter(img: ti.template()):
    sobel_x_filter_weights = ti.Matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    for I in ti.grouped(img):
        result = ti.Vector.zero(ti.f32, 3)
        for offset in ti.static(range(3)):
            # TODO: use ti.Vector.subscript
            result += img[I + ti.Vector([offset - 1, -1])][:3] * sobel_x_filter_weights[0, offset]
            # result += img[I - 1 + offset][:3] * sobel_x_filter_weights[offset]
        img[I][4:7] = result

@ti.kernel
def sobel_y_filter(img: ti.template()):
    sobel_y_filter_weights = ti.Matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    for I in ti.grouped(img):
        result = ti.Vector.zero(ti.f32, 3)
        for offset in ti.static(range(3)):
            result += img[I - 1 + offset][:3] * sobel_y_filter_weights[offset]
        img[I][7:10] = result

def percieve(state_grid: ti.template()):
    sobel_y_filter(state_grid)
    sobel_x_filter(state_grid)


def dense(input_vector: ti.template(), output_length: ti.i32):
    # convert input_vector to a 1D vector
    for i in range(output_length):
        result = ti.Vector.zero(ti.f32, 1)
        for j in range(input_vector.n):
            result += input_vector[j] * ti.random()
        input_vector[i] = result
    return input_vector


# simulate a relu activation
def relu(input_vector: ti.template()):
    for i in range(input_vector.n):
        if input_vector[i] < 0:
            input_vector[i] = 0
    return input_vector


def update_cell(cell: ti.template()):
    # simulate a dense neural network
    x = dense(cell, 128)
    x = relu(x)
    x = dense(x, 16)
    return x


@ti.kernel
def update(state_grid: ti.template()):
    for I in ti.grouped(state_grid):
        # convert to a vector
        state_grid[I] = update_cell(state_grid[I])


# read image into state_grid
img = ti.tools.imread('./res/example.jpg')
state_grid = ti.Vector.field(n=16, dtype=ti.f32, shape=(img.shape[0], img.shape[1]))
state_grid.from_numpy(img)
percieve(state_grid)
# display img channel first
ti.tools.imshow(state_grid.to_numpy()[:, :, :4])
ti.tools.imshow(state_grid.to_numpy()[:, :, 4:7])
ti.tools.imshow(state_grid.to_numpy()[:, :, 7:10])

update(state_grid)

# display img channel first
ti.tools.imshow(state_grid.to_numpy()[:, :, :4])
ti.tools.imshow(state_grid.to_numpy()[:, :, 4:7])
ti.tools.imshow(state_grid.to_numpy()[:, :, 7:10])