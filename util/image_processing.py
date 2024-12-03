import cv2
import numpy as np
import random
from threading import Thread

import platform

system_type = 'Linux'
if 'Windows' in platform.platform():
    system_type = 'Windows'

def imread(file_path,mod = 'normal',loadsize = 0, rgb=False):
    '''
    mod:  'normal' | 'gray' | 'all'
    loadsize: 0->original
    '''
    if system_type == 'Linux':
        if mod == 'normal':
            img = cv2.imread(file_path,1)
        elif mod == 'gray':
            img = cv2.imread(file_path,0)
        elif mod == 'all':
            img = cv2.imread(file_path,-1)
    
    #In windows, for chinese path, use cv2.imdecode insteaded.
    #It will loss EXIF, I can't fix it
    else: 
        if mod == 'normal':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),1)
        elif mod == 'gray':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),0)
        elif mod == 'all':
            img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            
    if loadsize != 0:
        img = resize(img, loadsize, interpolation=cv2.INTER_CUBIC)

    if rgb and img.ndim==3:
        img = img[:,:,::-1]

    return img

def imwrite(file_path,img,use_thread=False):
    '''
    in other to save chinese path images in windows,
    this fun just for save final output images
    '''
    def subfun(file_path,img):
        if system_type == 'Linux':
            cv2.imwrite(file_path, img)
        else:
            cv2.imencode('.png', img)[1].tofile(file_path)
    if use_thread:
        t = Thread(target=subfun,args=(file_path, img,))
        t.daemon()
        t.start
    else:
        subfun(file_path,img)

def resize(img: cv2.typing.MatLike, size, interpolation=cv2.INTER_LINEAR):
    '''
    cv2.INTER_NEAREST      Nearest-neighbor interpolation
    cv2.INTER_LINEAR       Bilinear interpolation
    cv2.INTER_AREA         Resampling with pixel neighborhood
    cv2.INTER_CUBIC        Bicubic interpolation with 4x4 sampling points
    cv2.INTER_LANCZOS4     Lanczos interpolation with an 8x8 pixel neighborhood
    '''
    h, w = img.shape[:2]
    if np.min((w,h)) ==size:
        return img
    if w >= h:
        res = cv2.resize(img, (int(size*w/h), size), interpolation=interpolation)
    else:
        res = cv2.resize(img, (size, int(size*h/w)), interpolation=interpolation)
    return res

def resize_like(img,img_like):
    h, w = img_like.shape[:2]
    img = cv2.resize(img, (w,h))
    return img

def ch_one2three(img):
    res = cv2.merge([img, img, img])
    return res

def color_adjust(img,alpha=0,beta=0,b=0,g=0,r=0,ran = False):
    '''
    g(x) = (1+α)g(x)+255*β, 
    g(x) = g(x[:+b*255,:+g*255,:+r*255])
    
    Args:
        img   : input image
        alpha : contrast
        beta  : brightness
        b     : blue hue
        g     : green hue
        r     : red hue
        ran   : if True, randomly generated color correction parameters
    Retuens:
        img   : output image
    '''
    img = img.astype('float')
    if ran:
        alpha = random.uniform(-0.1,0.1)
        beta  = random.uniform(-0.1,0.1)
        b     = random.uniform(-0.05,0.05)
        g     = random.uniform(-0.05,0.05)
        r     = random.uniform(-0.05,0.05)
    img = (1+alpha)*img+255.0*beta
    bgr = [b*255.0,g*255.0,r*255.0]
    for i in range(3): img[:,:,i]=img[:,:,i]+bgr[i]
    
    return (np.clip(img,0,255)).astype('uint8')

def CAdaIN(src,dst):
    '''
    make src has dst's style
    '''
    return np.std(dst)*((src-np.mean(src))/np.std(src))+np.mean(dst)

def makedataset(target_image,orgin_image):
    target_image = resize(target_image,256)
    orgin_image = resize(orgin_image,256)
    img = np.zeros((256,512,3), dtype = "uint8")
    w = orgin_image.shape[1]
    img[0:256,0:256] = target_image[0:256,int(w/2-256/2):int(w/2+256/2)]
    img[0:256,256:512] = orgin_image[0:256,int(w/2-256/2):int(w/2+256/2)]
    return img

def find_mostlikely_ROI(mask):
    contours,hierarchy=cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        areas = []
        for contour in contours:
            areas.append(cv2.contourArea(contour))
        index = areas.index(max(areas))
        mask = np.zeros_like(mask)
        mask = cv2.fillPoly(mask,[contours[index]],(255))
    return mask

def boundingSquare(mask, Ex_mul):
    """
    Calculate the center, bounding box size, and expansion of the mosaic mask region.
    """
    # Calculate the region area
    area = cv2.countNonZero(mask)
    if area == 0:
        return 0, 0, 0, 0

    # Call the encapsulated bounding box calculation function
    h, w = mask.shape[:2]
    x, y, size = compute_bounding_box(mask, Ex_mul, h, w)

    # Call the coordinate adjustment function to ensure the result is within image bounds
    x, y = adjust_coordinates(x, y, size, h, w)

    # Return the result
    half_size = size // 2
    return x + half_size, y + half_size, half_size, area

def compute_bounding_box(mask, Ex_mul, h, w):
    """
    Calculate the expanded bounding box of the mask and adjust it based on Ex_mul.
    """
    # Calculate the bounding box of the mask
    x, y, width, height = cv2.boundingRect(mask)
    center_x = x + width // 2
    center_y = y + height // 2

    # Determine the size of the bounding box
    size = max(width, height) * Ex_mul
    size = min(size, min(h, w))  # Ensure it does not exceed the image dimensions

    return center_x - size // 2, center_y - size // 2, int(size)

def adjust_coordinates(x, y, size, h, w):
    """
    Adjust the bounding box position to ensure it is within the image boundaries.
    """
    x = np.clip(x, 0, w - size)
    y = np.clip(y, 0, h - size)
    return x, y

def boundingSquare_old(mask,Ex_mul):
    # thresh = mask_threshold(mask,10,threshold)
    area = mask_area(mask)
    if area == 0 :
        return 0,0,0,0

    x,y,w,h = cv2.boundingRect(mask)
    
    center = np.array([int(x+w/2),int(y+h/2)])
    size = max(w,h)
    point0=np.array([x,y])
    point1=np.array([x+size,y+size])

    h, w = mask.shape[:2]
    if size*Ex_mul > min(h, w):
        size = min(h, w)
        halfsize = int(min(h, w)/2)
    else:
        size = Ex_mul*size
        halfsize = int(size/2)
        size = halfsize*2
    point0 = center - halfsize
    point1 = center + halfsize
    if point0[0]<0:
        point0[0]=0
        point1[0]=size
    if point0[1]<0:
        point0[1]=0
        point1[1]=size
    if point1[0]>w:
        point1[0]=w
        point0[0]=w-size
    if point1[1]>h:
        point1[1]=h
        point0[1]=h-size
    center = ((point0+point1)/2).astype('int')
    return center[0],center[1],halfsize,area

def mask_threshold(mask,ex_mun,threshold):
    mask = cv2.threshold(mask,threshold,255,cv2.THRESH_BINARY)[1]
    mask = cv2.blur(mask, (ex_mun, ex_mun))
    mask = cv2.threshold(mask,threshold/5,255,cv2.THRESH_BINARY)[1]
    return mask

def mask_area(mask):
    mask = cv2.threshold(mask,127,255,0)[1]
    # contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1] #for opencv 3.4
    contours= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]#updata to opencv 4.0
    try:
        area = cv2.contourArea(contours[0])
    except:
        area = 0
    return area

def replace_mosaic(img_origin, img_fake, mask, x, y, size, no_feather, enable_dynamic_blur=False, base_ksize=5, gradient_scale=1.0):
    # Resize the mosaic area
    img_fake = cv2.resize(img_fake, (size * 2, size * 2), interpolation=cv2.INTER_CUBIC)
    
    if no_feather:
        # Directly replace without transition processing
        img_origin[y - size:y + size, x - size:x + size] = img_fake
        return img_origin
    else:
        # Dynamic blur
        if enable_dynamic_blur:
            eclosion_num = dynamic_blur(mask, mask, base_ksize=base_ksize, gradient_scale=gradient_scale)
        else:
            eclosion_num = int(size / 10) + 2  # Default blur kernel size

        # Pre-scale and crop the mask
        mask_crop = cv2.resize(mask, (img_origin.shape[1], img_origin.shape[0]))[y - size:y + size, x - size:x + size]
        mask_crop = cv2.merge([mask_crop, mask_crop, mask_crop])  # Add channel dimensions

        # Apply blur to the mask
        mask_crop = cv2.blur(mask_crop, (eclosion_num, eclosion_num)) if not enable_dynamic_blur else mask_crop
        mask_crop = mask_crop / 255.0
        
        # Composite the image
        img_crop = img_origin[y - size:y + size, x - size:x + size]
        img_origin[y - size:y + size, x - size:x + size] = np.clip(
            (img_crop * (1 - mask_crop) + img_fake * mask_crop), 0, 255
        ).astype('uint8')

        return img_origin

def replace_mosaic_1(img_origin, img_fake, mask, x, y, size, no_feather):
    img_fake = cv2.resize(img_fake, (size * 2, size * 2), interpolation=cv2.INTER_CUBIC)
    if no_feather:
        img_origin[y - size:y + size, x - size:x + size] = img_fake
        return img_origin
    
    # Pre-scale the mask and crop it
    mask_crop = mask[y - size:y + size, x - size:x + size]
    mask_crop = cv2.blur(mask_crop, (int(size / 10) + 2, int(size / 10) + 2))
    mask_crop = mask_crop.astype(np.float32) / 255.0
    
    # Blend the images
    mask_crop = mask_crop[..., None]  # Add a channel dimension to avoid repeatedly using ch_one2three
    img_crop = img_origin[y - size:y + size, x - size:x + size].astype(np.float32)
    img_fake = img_fake.astype(np.float32)
    img_origin[y - size:y + size, x - size:x + size] = np.clip(
        img_crop * (1 - mask_crop) + img_fake * mask_crop, 0, 255
    ).astype('uint8')
    
    return img_origin

def replace_mosaic_old(img_origin,img_fake,mask,x,y,size,no_feather):
    img_fake = cv2.resize(img_fake,(size*2,size*2),interpolation=cv2.INTER_CUBIC)
    if no_feather:
        img_origin[y-size:y+size,x-size:x+size]=img_fake
        return img_origin
    else:
        # #color correction
        # RGB_origin = img_origin[y-size:y+size,x-size:x+size].mean(0).mean(0)
        # RGB_fake = img_fake.mean(0).mean(0)
        # for i in range(3):img_fake[:,:,i] = np.clip(img_fake[:,:,i]+RGB_origin[i]-RGB_fake[i],0,255)
        #eclosion
        eclosion_num = int(size/10)+2

        mask_crop = cv2.resize(mask,(img_origin.shape[1],img_origin.shape[0]))[y-size:y+size,x-size:x+size]
        mask_crop = ch_one2three(mask_crop)

        mask_crop = (cv2.blur(mask_crop, (eclosion_num, eclosion_num)))
        mask_crop = mask_crop/255.0

        img_crop = img_origin[y-size:y+size,x-size:x+size]
        img_origin[y-size:y+size,x-size:x+size] = np.clip((img_crop*(1-mask_crop)+img_fake*mask_crop),0,255).astype('uint8')
        
        return img_origin


def Q_lapulase(resImg):
    '''
    Evaluate image quality
    score > 20   normal
    score > 50   clear
    '''
    img2gray = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
    img2gray = resize(img2gray,512)
    res = cv2.Laplacian(img2gray, cv2.CV_64F)
    score = res.var()
    return score

def psnr(img1,img2):
    mse = np.mean((img1/255.0-img2/255.0)**2)
    if mse < 1e-10:
        return 100
    psnr_v = 20*np.log10(1/np.sqrt(mse))
    return psnr_v

def splice(imgs,splice_shape):
    '''Stitching multiple images, all imgs must have the same size
    imgs : [img1,img2,img3,img4]
    splice_shape: (2,2)
    '''
    h,w,ch = imgs[0].shape
    output = np.zeros((h*splice_shape[0],w*splice_shape[1],ch),np.uint8)
    cnt = 0
    for i in range(splice_shape[0]):
        for j in range(splice_shape[1]):
            if cnt < len(imgs):
                output[h*i:h*(i+1),w*j:w*(j+1)] = imgs[cnt]
                cnt += 1
    return output

def dynamic_blur(mask, image, base_ksize=5, gradient_scale=1.0):
    # Calculate the average gradient
    avg_gradient = calculate_average_gradient(mask)

    # Dynamically adjust blur parameters
    ksize = max(base_ksize, int(avg_gradient * gradient_scale))  # Adjust the blur kernel size based on the gradient
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure the kernel size is an odd number

    # Apply blurring
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image

def calculate_average_gradient(mask):
    # Find the mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_mask = np.zeros_like(mask)
    cv2.drawContours(boundary_mask, contours, -1, 255, thickness=1)  # Draw the boundary

    # Compute gradients
    gradient_magnitude = calculate_gradient(mask)

    # Extract gradient values only in the boundary region
    boundary_gradient = gradient_magnitude[boundary_mask == 255]

    # Calculate the average gradient
    average_gradient = np.mean(boundary_gradient) if len(boundary_gradient) > 0 else 0
    return average_gradient

def calculate_gradient(mask):
    # Use Sobel operator to compute gradients
    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in the x-direction
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in the y-direction
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)  # Gradient magnitude
    return gradient_magnitude
