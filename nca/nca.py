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

# read image into state_grid
img = ti.tools.imread('./res/example.jpg')
state_grid = ti.Vector.field(n=16, dtype=ti.f32, shape=(img.shape[0], img.shape[1]))
state_grid.from_numpy(img)
print(img[0,0])
percieve(state_grid)
# display img channel first
ti.tools.imshow(state_grid.to_numpy()[:, :, :4])
ti.tools.imshow(state_grid.to_numpy()[:, :, 4:7])
ti.tools.imshow(state_grid.to_numpy()[:, :, 7:10])

