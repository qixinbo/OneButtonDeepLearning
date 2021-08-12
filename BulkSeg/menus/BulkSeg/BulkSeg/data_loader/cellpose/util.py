import numpy as np
from time import time
import random, math, itertools
import scipy.ndimage as ndimg
# from .edt import distance_transform_edt
from .findmax import find_maximum
from scipy.ndimage import distance_transform_edt, maximum_filter

def msk2flow(msk):
    idx = np.where(maximum_filter(msk, size=3) > msk)
    msk[idx] = 0

    # dist = ndimg.distance_transform_edt(msk)
    # np.save("dist.npy", dist)

    # pts = find_maximum(dist, 1, True)
    # # print("pts = ", pts)
    # dist[pts[:, 0], pts[:, 1]] += 1

    objs = ndimg.find_objects(msk)

    signal = np.zeros(msk.shape+(2, ))

    pts = []

    for i, obj in enumerate(objs):
        if obj is not None:
            objr, objc = obj
            height, width = objr.stop - objr.start + 1, objc.stop - objc.start + 1
            rows_in_obj, cols_in_obj = np.nonzero(msk[objr, objc] == (i+1))

            rows_in_obj = rows_in_obj.astype(np.int32) + 1
            cols_in_obj = cols_in_obj.astype(np.int32) + 1
            median_row = np.median(rows_in_obj)
            median_col = np.median(cols_in_obj)
            imin = np.argmin((cols_in_obj-median_col)**2 + (rows_in_obj-median_row)**2)
            median_row = rows_in_obj[imin]
            median_col = cols_in_obj[imin]

            signal[objr.start+rows_in_obj-1, objc.start+cols_in_obj-1, 0] = median_row - rows_in_obj
            signal[objr.start+rows_in_obj-1, objc.start+cols_in_obj-1, 1] = median_col - cols_in_obj

            pts.append([objr.start+median_row-1, objc.start+median_col-1])

    # # Plot the centroids
    # print("pts = ", pts)
    # import matplotlib.pyplot as plt
    # plt.imshow(msk, cmap='gray')
    # plt.plot([i[1] for i in pts], [i[0] for i in pts], 'b.')
    # plt.show()

    shp = signal.shape[:-1]
    l = np.linalg.norm(signal, axis=-1)
    signal /= l.reshape(shp+(1,))+1e-20

    return signal

    # # k0 = np.array([[1,-1],[2,-2],[1,-1]])
    # # k1 = np.array([[1,2,1],[-1,-2,-1]])
    # k0 = np.array([[2, 1, 0],[1, 0, -1],[0, -1,-2]])
    # k1 = np.array([[0,1,2], [-1, 0, 1], [-2,-1,0]])
    # sobel0 = ndimg.convolve(dist, k1)
    # sobel1 = ndimg.convolve(dist, k0)
    # # sobel1 = ndimg.convolve(dist, [[1,-1]])
    # # sobel0 = ndimg.convolve(dist, [[1],[-1]])
    # # sobel0 = ndimg.gaussian_filter(sobel0, 1)
    # # sobel1 = ndimg.gaussian_filter(sobel1, 1)
    # print("sobel0.shape = ", sobel0.shape)
    # sobel = np.stack((sobel0, sobel1), axis=-1)
    # print("sobel.shape = ", sobel.shape)

    # # sobel0 = ndimg.sobel(dist, 0, output=dist.dtype)
    # # sobel1 = ndimg.sobel(dist, 1, output=dist.dtype)
    # # sobel = np.stack((sobel0, sobel1), axis=-1)

    # # shp, dim = sobel.shape[:-1], sobel.ndim - 1
    # # l = np.linalg.norm(sobel, axis=-1)
    # # sobel /= l.reshape(shp+(1,))+1e-20


    # return sobel


def sigmoid_func(prob):
    prob = prob.copy()
    prob*=-1; np.exp(prob, out=prob);
    prob+=1; np.divide(1, prob, out=prob)
    return prob

def estimate_volumes(arr, sigma=3):
    msk = arr > 50; 
    idx = np.arange(len(arr), dtype=np.uint32)
    idx, arr = idx[msk], arr[msk]
    for k in np.linspace(5, sigma, 5):
       std = arr.std()
       dif = np.abs(arr - arr.mean())
       msk = dif < std * k
       idx, arr = idx[msk], arr[msk]
    return arr.mean(), arr.std()

def flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None):
    flowp = np.asarray(flowp)
    shp, dim = flowp.shape[:-1], flowp.ndim - 1
    l = np.linalg.norm(flowp[:,:,:2], axis=-1)
    flow = flowp[:,:,:2]/(l.reshape(shp+(1,))+1.0e-9)
    flow[(flowp[:,:,2]<level)|(l<grad)] = 0
    ss = ((slice(None),) * (dim) + ([0,-1],)) * 2
    for i in range(dim):flow[ss[dim-i:-i-2]+(i,)]=0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1, dim)
    strides = np.cumprod(np.array((1,)+shp[::-1]))
    dn = (strides[-2::-1] * dn).sum(axis=-1)

    rst = np.arange(flow.size//dim); rst += dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, None, len(rst))
    hist = hist.astype(np.uint32).reshape(shp)
    lab, n = ndimg.label(hist, np.ones((3,)*dim))
    volumes = ndimg.sum(hist, lab, np.arange(n+1))
    areas = np.bincount(lab.ravel())
    mean, std = estimate_volumes(volumes, 2)
    if not volume: volume = max(mean-std*3, 50)
    if not area: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)

#from concurrent.futures import ThreadPoolExecutor
def make_slice(l, w, mar):
    r = np.linspace(w//2, l-w//2, math.ceil((l-mar)/(w-mar))).astype(int)
    return [slice(i-w//2, i+w//2) for i in r.tolist()]

def grid_slice(H, W, size, mar):
    a, b = make_slice(H, size, mar), make_slice(W, size, mar)
    return list(itertools.product(a, b))


def pad(img, shp, mode='constant', constant_values=0):
    if shp[2][0]==shp[2][1]==shp[3][0]==shp[3][1]==0: return img
    (n, c, h, w), (mn, mc, mh, mw) = img.shape, shp
    newimg = np.zeros((n, c, h+mh[0]*2, w+mw[0]*2), dtype=img.dtype)
    newimg[:,:,mh[0]:-mh[1],mw[0]:-mw[1]] = img
    return newimg

def conv(img, core, group=1, stride=(1, 1), dilation=(1, 1)):
    #threadPool = ThreadPoolExecutor(max_workers=1)
    (strh, strw), (dh, dw) = stride, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    cimg_w = c * h * w * group
    cimg_h, i = (hi//strh)*(wi//strw), 0
    shp = ((0, 0), (0, 0), (dh*(h//2),)*2, (dw*(w//2),)*2)
    img = pad(img, shp, 'constant', constant_values=0)
    img = img.transpose((1,0,2,3)) # nchw -> cnhw
    col_img = np.zeros((ci, w*h,  ni, hi//strh, wi//strw), img.dtype) #(h*w, c, N, H, W)
    #def set_value(img, i, v): img[:,i] = v
    for r in range(0, h*dh, dh):
        for c in range(0, w*dw, dw):
            col_img[:,i], i = img[:,:,0+r:hi+r:strh, 0+c:wi+c:strw], i+1
            #threadPool.submit(set_value, col_img, i-1, im)
    #threadPool.shutdown(wait=True)
    col_core = core.reshape((group, core.shape[0]//group, -1))
    col_img.shape = (group, cimg_w//group, -1)
    rst = [i.dot(j) for i, j in zip(col_core, col_img)]
    rst = rst[0] if group==1 else np.concatenate(rst)
    return rst.reshape((n, ni, hi//strh, wi//strw)).transpose(1, 0, 2, 3)

def pool_nxn(img, f, s):
    n, c, h, w = img.shape
    rshp = img.reshape(n,c,h//s,s,w//s,s)
    rshp = rshp.transpose((0,1,2,4,3,5))
    if f == 'max': return rshp.max(axis=(4,5))
    if f == 'mean': return rshp.mean(axis=(4,5))

def pool(img, f, core=(2, 2), stride=(2, 2)):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), ((ch-1)//2,)*2, ((cw-1)//2,)*2)
    img = pad(img, shp, 'constant', constant_values=0)
    (imn, ic, ih, iw), imgs = img.shape, []
    buf = np.zeros(img.shape[:2]+(h//strh,w//strw), np.float32)
    buf -= 1e4
    for r in range(0, ch, 1):
        for c in range(0, cw, 1):
            f(img[:,:,r:h+r:strh,c:w+c:strw], buf, out=buf)
    return buf


def maxpool(i, c=(2, 2), s=(2, 2)):return pool(i, np.maximum, c, s)

def avgpool(i, c=(2, 2), s=(2, 2)): return pool(i, 'mean', c, s)
    
def resize(img, size):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    kh, kw = size[0]/h, size[1]/w
    slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
    rs = np.linspace(*slicer, dtype=np.float32)
    slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
    cs = np.linspace(*slicec, dtype=np.float32)
    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    ra = np.floor(np.clip(rs, 0, h-1.5))
    ca = np.floor(np.clip(cs, 0, w-1.5))
    ra, ca = ra.astype(int), ca.astype(int)
    rs -= ra; cs -= ca; rb = ra+1; cb = ca+1;
    rs.shape, img.shape = (-1,1), (-1, h, w)
    buf = img[:,:,ca]*(1-cs) + img[:,:,cb]*cs
    result = buf[:,ra,:]*(1-rs) + buf[:,rb,:]*rs
    return result.reshape(nc + size)

def make_upmat(k):
    xs = np.linspace(0.5/k, 1-0.5/k, k*1, dtype=np.float32)
    rs, cs = xs[:,None], xs[None,:]
    klt = ((1-cs)*(1-rs)).reshape((1,-1))
    krt = (cs * (1-rs)).reshape((1,-1))
    klb = ((1-cs) * rs).reshape((1,-1))
    krb = (cs * rs).reshape((1,-1))
    return np.vstack([klt, krt, klb, krb])
    
def upsample_blinear(img, k, matbuf={}):    
    n, c, h, w = img.shape
    img = (img[:,:,:1,:], img, img[:,:,-1:,:])
    img = np.concatenate(img, axis=2)
    img = (img[:,:,:,:1], img, img[:,:,:,-1:])
    img = np.concatenate(img, axis=3)
    if not k in matbuf: matbuf[k] = make_upmat(k)    
    imgs = [img[:,:,:-1,:-1], img[:,:,:-1,1:],
            img[:,:,1:,:-1], img[:,:,1:,1:]]
    imgs = [i[:,:,:,:,None] for i in imgs]
    rst = np.concatenate(imgs, axis=-1)
    rst = np.dot(rst.reshape((-1,4)), matbuf[k])
    rst = rst.reshape((-1, w+1, k, k))
    rst = rst.transpose((0,2,1,3))
    rst = rst.reshape((n,c,(h+1)*k, (w+1)*k))
    return rst[:,:,k//2:-k//2,k//2:-k//2]

def upsample_nearest(img, k):
    n, c, h, w = img.shape
    rst = np.zeros((n, c, h, k, w, k), dtype=np.float32)
    trst = rst.transpose((0,1,2,4,3,5))
    trst[:] = img[:,:,:,:,None,None]
    return rst.reshape((n, c, h*k, w*k))

def upsample(img, k, mode):
    if mode=='nearest': return upsample_nearest(img, k)
    if mode=='linear': return upsample_blinear(img, k)


if __name__ == '__main__':
    img = np.zeros((1, 64, 512, 512), dtype=np.float32)
    core = np.zeros((32, 64, 3, 3), dtype=np.float32)

    conv(img, core)
