import numpy as np
from numba import njit, prange

@njit(inline="always")
def _clamp_int(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value

@njit(parallel=True, fastmath=True)
def write_nearest(volume, data, origin, normal_axis, a0, a1, d0, dd, y0, dy, x0, dx):

    Z, Y, X = volume.shape
    D, H, W = data.shape

    inv_dd = 0.0 if dd == 0.0 else 1.0 / dd
    inv_dy = 0.0 if dy == 0.0 else 1.0 / dy
    inv_dx = 0.0 if dx == 0.0 else 1.0 / dx

    d_halfwidth = 0.5

    for z in prange(Z):
        gz = float(z)
        for y in range(Y):
            gy = float(y)
            for x in range(X):
                gx = float(x)

                rz = gz - origin[0]
                ry = gy - origin[1]
                rx = gx - origin[2]

                d  = rz*normal_axis[0] + ry*normal_axis[1] + rx*normal_axis[2]
                yy = rz*a0[0]          + ry*a0[1]          + rx*a0[2]
                xx = rz*a1[0]          + ry*a1[1]          + rx*a1[2]

                # depth index
                if D == 1:
                    # Reject voxels far from the intended slice plane
                    if np.abs(d - d0) > d_halfwidth:
                        continue
                    di = 0
                else:
                    di = int(np.rint((d - d0) * inv_dd))
                    if di < 0 or di >= D:
                        continue

                # y/x indices
                if H == 1:
                    hi = 0
                else:
                    hi = int(np.rint((yy - y0) * inv_dy))

                if W == 1:
                    wi = 0
                else:
                    wi = int(np.rint((xx - x0) * inv_dx))

                if hi < 0 or hi >= H or wi < 0 or wi >= W:
                    continue

                val = data[di, hi, wi]
                if val != 0:
                    volume[z, y, x] = val

@njit(parallel=True, fastmath=True)
def read_nearest(volume, output, origin, normal_axis, a0, a1, d0, dd, y0, dy, x0, dx):

    Z, Y, X = volume.shape
    D, H, W = output.shape
    zmax = Z - 1
    ymax = Y - 1
    xmax = X - 1

    for di in prange(D):
        d = d0 + di * dd
        for hi in range(H):
            y = y0 + hi * dy
            for wi in range(W):
                x = x0 + wi * dx

                gz = origin[0] + d*normal_axis[0] + y*a0[0] + x*a1[0]
                gy = origin[1] + d*normal_axis[1] + y*a0[1] + x*a1[1]
                gx = origin[2] + d*normal_axis[2] + y*a0[2] + x*a1[2]

                iz = _clamp_int(int(np.rint(gz)), 0, zmax)
                iy = _clamp_int(int(np.rint(gy)), 0, ymax)
                ix = _clamp_int(int(np.rint(gx)), 0, xmax)

                output[di, hi, wi] = volume[iz, iy, ix]

@njit(inline="always")
def _linear_interpolate(a, b, t):
    return a + t * (b - a)

@njit(parallel=True, fastmath=True)
def read_trilinear(volume, output, origin, normal_axis, a0, a1, d0, dd, y0, dy, x0, dx):

    Z, Y, X = volume.shape
    D, H, W = output.shape
    zmax = Z - 1
    ymax = Y - 1
    xmax = X - 1

    for di in prange(D):
        d = d0 + di * dd
        for hi in range(H):
            y = y0 + hi * dy
            for wi in range(W):
                x = x0 + wi * dx

                gz = origin[0] + d*normal_axis[0] + y*a0[0] + x*a1[0]
                gy = origin[1] + d*normal_axis[1] + y*a0[1] + x*a1[1]
                gx = origin[2] + d*normal_axis[2] + y*a0[2] + x*a1[2]

                z0i = int(np.floor(gz))
                y0i = int(np.floor(gy))
                x0i = int(np.floor(gx))

                tz = gz - z0i
                ty = gy - y0i
                tx = gx - x0i

                z0c = _clamp_int(z0i, 0, zmax)
                y0c = _clamp_int(y0i, 0, ymax)
                x0c = _clamp_int(x0i, 0, xmax)
                z1c = _clamp_int(z0i + 1, 0, zmax)
                y1c = _clamp_int(y0i + 1, 0, ymax)
                x1c = _clamp_int(x0i + 1, 0, xmax)

                c000 = volume[z0c, y0c, x0c]
                c001 = volume[z0c, y0c, x1c]
                c010 = volume[z0c, y1c, x0c]
                c011 = volume[z0c, y1c, x1c]
                c100 = volume[z1c, y0c, x0c]
                c101 = volume[z1c, y0c, x1c]
                c110 = volume[z1c, y1c, x0c]
                c111 = volume[z1c, y1c, x1c]

                c00 = _linear_interpolate(c000, c001, tx)
                c01 = _linear_interpolate(c010, c011, tx)
                c10 = _linear_interpolate(c100, c101, tx)
                c11 = _linear_interpolate(c110, c111, tx)

                c0 = _linear_interpolate(c00, c01, ty)
                c1 = _linear_interpolate(c10, c11, ty)

                output[di, hi, wi] = _linear_interpolate(c0, c1, tz)
