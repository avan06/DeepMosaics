import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from models import runmodel
from util import data,util,ffmpeg,filt
from util import image_processing as impro
from .init import video_init
from multiprocessing import Queue, Process
from threading import Thread

'''
---------------------Clean Mosaic---------------------
'''
def get_mosaic_positions(opt, netM: nn.Module, imagepaths: list[str], savemask=True):
    positions = []
    resume_frame = 0
    continue_flag = False
    
    # Step 1: Resume Check
    step, pre_positions = resume_check(opt)
    if step and step['step'] > 2:
        return pre_positions
    if step and step['step'] >= 2 and step['frame'] > 0:
        resume_frame = step['frame']
        continue_flag = True
        positions = pre_positions.tolist()
        imagepaths = imagepaths[resume_frame:]

    print('Step: 2/4 -- Find mosaic locations')
    start_time = time.time()

    # Step 2: Setup Thread for Image Loading
    img_read_pool = Queue(maxsize=4)
    Thread(target=image_loader, args=(imagepaths, img_read_pool, opt.temp_dir)).start()

    # Step 3: Process Images
    for i, imagepath in enumerate(imagepaths, start=1):
        img_origin = img_read_pool.get()
        x, y, size, mask = runmodel.get_mosaic_position(img_origin, netM, opt)
        positions.append([x, y, size])

        # Save mask in a thread
        if savemask:
            Thread(target=cv2.imwrite, args=(os.path.join(opt.temp_dir, 'mosaic_mask', imagepath), mask)).start()

        # Periodic Save
        if i % 1000 == 0:
            save_progress(positions, opt.temp_dir, step, continue_flag, pre_positions, i + resume_frame)

        # Preview Result
        if not opt.no_preview:
            preview_mask(mask)

        # Display Progress
        print_progress(i, len(imagepaths), start_time)

    # Step 4: Cleanup
    finalize_preview(opt.no_preview)
    print("\nOptimize mosaic locations...")
    positions = finalize_positions(positions, pre_positions, continue_flag, opt.medfilt_num)
    save_final_step(positions, opt.temp_dir)

    return positions

# Helper Functions
def resume_check(opt):
    step = None
    pre_positions = None
    step_path = os.path.join(opt.temp_dir, 'step.json')
    positions_path = os.path.join(opt.temp_dir, 'mosaic_positions.npy')

    if os.path.isfile(step_path):
        step = util.loadjson(step_path)
        if os.path.isfile(positions_path):
            pre_positions = np.load(positions_path)
    return step, pre_positions

def image_loader(imagepaths, img_read_pool, temp_dir):
    for imagepath in imagepaths:
        img_origin = impro.imread(os.path.join(temp_dir, 'video2image', imagepath))
        img_read_pool.put(img_origin)

def save_progress(positions, temp_dir, step, continue_flag, pre_positions, frame):
    save_positions = np.array(positions)
    if continue_flag:
        save_positions = np.concatenate((pre_positions, save_positions), axis=0)
    np.save(os.path.join(temp_dir, 'mosaic_positions.npy'), save_positions)
    util.savejson(os.path.join(temp_dir, 'step.json'), {'step': 2, 'frame': frame})

def preview_mask(mask):
    cv2.imshow('mosaic mask', mask)
    cv2.waitKey(1)

def print_progress(current, total, start_time):
    elapsed_time = time.time() - start_time
    print(f'\r {current}/{total} | Elapsed: {elapsed_time:.2f}s', end='')

def finalize_preview(no_preview):
    if not no_preview:
        cv2.destroyAllWindows()

def finalize_positions(positions, pre_positions, continue_flag, medfilt_num):
    positions = np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions, positions), axis=0)
    for i in range(3):
        positions[:, i] = filt.medfilt(positions[:, i], medfilt_num)
    return positions


def save_final_step(positions, temp_dir):
    util.savejson(os.path.join(temp_dir, 'step.json'), {'step': 3, 'frame': 0})
    np.save(os.path.join(temp_dir, 'mosaic_positions.npy'), positions)

def get_mosaic_positions_old(opt, netM: nn.Module, imagepaths: list[str], savemask=True):
    # resume
    continue_flag = False
    if os.path.isfile(os.path.join(opt.temp_dir,'step.json')):
        step = util.loadjson(os.path.join(opt.temp_dir,'step.json'))
        resume_frame = int(step['frame'])
        if int(step['step'])>2:
            pre_positions = np.load(os.path.join(opt.temp_dir,'mosaic_positions.npy'))
            return pre_positions
        if int(step['step'])>=2 and resume_frame>0:
            pre_positions = np.load(os.path.join(opt.temp_dir,'mosaic_positions.npy'))
            continue_flag = True
            imagepaths = imagepaths[resume_frame:]
            
    positions = []
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('mosaic mask', cv2.WINDOW_NORMAL)
    print('Step:2/4 -- Find mosaic location')

    img_read_pool = Queue(4)
    def loader(imagepaths):
        for imagepath in imagepaths:
            img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
            img_read_pool.put(img_origin)
    t = Thread(target=loader,args=(imagepaths,))
    t.setDaemon(True)
    t.start()

    for i,imagepath in enumerate(imagepaths,1):
        img_origin = img_read_pool.get()
        x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
        positions.append([x,y,size])
        if savemask:
            t = Thread(target=cv2.imwrite, args=(os.path.join(opt.temp_dir+'/mosaic_mask', imagepath), mask,))
            t.start()
        if i%1000==0:
            save_positions = np.array(positions)
            if continue_flag:
                save_positions = np.concatenate((pre_positions,save_positions),axis=0)
            np.save(os.path.join(opt.temp_dir,'mosaic_positions.npy'),save_positions)
            step = {'step':2,'frame':i+resume_frame}
            util.savejson(os.path.join(opt.temp_dir,'step.json'),step)

        #preview result and print
        if not opt.no_preview:
            cv2.imshow('mosaic mask',mask)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i)+'/'+str(len(imagepaths)),util.get_bar(100*i/len(imagepaths),num=35),util.counttime(t1,t2,i,len(imagepaths)),end='')
    
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('\nOptimize mosaic locations...')
    positions =np.array(positions)
    if continue_flag:
        positions = np.concatenate((pre_positions,positions),axis=0)
    for i in range(3):positions[:,i] = filt.medfilt(positions[:,i],opt.medfilt_num)
    step = {'step':3,'frame':0}
    util.savejson(os.path.join(opt.temp_dir,'step.json'),step)
    np.save(os.path.join(opt.temp_dir,'mosaic_positions.npy'),positions)

    return positions

def cleanmosaic_img(opt, netG:nn.Module, netM:nn.Module):
    path = opt.media_path
    print('Clean Mosaic:', path)
    img_origin:cv2.typing.MatLike = impro.imread(path)
    
    # Step 1: Retrieve all mosaic regions
    mosaic_regions = runmodel.get_all_mosaic_positions(img_origin, netM, opt)  # Returns multiple (x, y, size, mask)
    img_result = img_origin.copy()

    # Step 2: Process each region one by one
    for x, y, size, mask in mosaic_regions:
        if size > 100:  # Ensure the region size is sufficient for processing
            img_mosaic = img_origin[y - size:y + size, x - size:x + size]
            
            if opt.traditional:
                img_fake = runmodel.traditional_cleaner(img_mosaic, opt)
            else:
                img_fake = runmodel.run_pix2pix(img_mosaic, netG, opt)
            
            # Replace the current mosaic region
            img_result = impro.replace_mosaic(img_result, img_fake, mask, x, y, size, opt.no_feather)
        else:
            print(f'Skipped mosaic region at ({x}, {y}) with insufficient size({size})')
    
    # Step 3: Save the result image
    output_path = os.path.join(opt.result_dir, os.path.splitext(os.path.basename(path))[0] + '_clean.png')
    impro.imwrite(output_path, img_result)

def cleanmosaic_img_old(opt,netG:nn.Module,netM:nn.Module):
    path = opt.media_path
    print('Clean Mosaic:',path)
    img_origin = impro.imread(path)
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    #cv2.imwrite('./mask/'+os.path.basename(path), mask)
    img_result = img_origin.copy()
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    else:
        print('Do not find mosaic')
    impro.imwrite(os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.png'),img_result)

def cleanmosaic_img_server(opt,img_origin,netG:nn.Module,netM:nn.Module):
    x,y,size,mask = runmodel.get_mosaic_position(img_origin,netM,opt)
    img_result = img_origin.copy()
    if size > 100 :
        img_mosaic = img_origin[y-size:y+size,x-size:x+size]
        if opt.traditional:
            img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
        else:
            img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
        img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
    return img_result

def cleanmosaic_video_byframe(opt,netG:nn.Module,netM:nn.Module):
    path = opt.media_path
    fps,imagepaths,height,width = video_init(opt,path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame-1):]

    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)

    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        img_origin = impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepath))
        img_result = img_origin.copy()
        if size > 100:
            try:#Avoid unknown errors
                img_mosaic = img_origin[y-size:y+size,x-size:x+size]
                if opt.traditional:
                    img_fake = runmodel.traditional_cleaner(img_mosaic,opt)
                else:
                    img_fake = runmodel.run_pix2pix(img_mosaic,netG,opt)
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            except Exception as e:
                print('Warning:',e)
        t = Thread(target=cv2.imwrite,args=(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath), img_result,))
        t.start()
        os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
        
        #preview result and print
        if not opt.no_preview:
            cv2.imshow('clean',img_result)
            cv2.waitKey(1) & 0xFF
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4'))  

def cleanmosaic_video_fusion(opt,netG:nn.Module,netM:nn.Module):
    path = opt.media_path
    N,T,S = 2,5,3
    LEFT_FRAME = (N*S)
    POOL_NUM = LEFT_FRAME*2+1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S,T,dtype=np.int64)
    img_pool = []
    previous_frame = None
    init_flag = True
    
    fps,imagepaths,height,width = video_init(opt,path)
    start_frame = int(imagepaths[0][7:13])
    positions = get_mosaic_positions(opt, netM, imagepaths, savemask=True)[(start_frame-1):]
    t1 = time.time()
    if not opt.no_preview:
        cv2.namedWindow('clean', cv2.WINDOW_NORMAL)
    
    # clean mosaic
    print('Step:3/4 -- Clean Mosaic:')
    length = len(imagepaths)
    write_pool = Queue(4)
    show_pool = Queue(4)
    def write_result():
        while True:
            save_ori,imagepath,img_origin,img_fake,x,y,size = write_pool.get()
            if save_ori:
                img_result = img_origin
            else:
                mask = cv2.imread(os.path.join(opt.temp_dir+'/mosaic_mask',imagepath),0)
                img_result = impro.replace_mosaic(img_origin,img_fake,mask,x,y,size,opt.no_feather)
            if not opt.no_preview:
                show_pool.put(img_result.copy())
            cv2.imwrite(os.path.join(opt.temp_dir+'/replace_mosaic',imagepath),img_result)
            os.remove(os.path.join(opt.temp_dir+'/video2image',imagepath))
    t = Thread(target=write_result,args=())
    t.setDaemon(True)
    t.start()

    for i,imagepath in enumerate(imagepaths,0):
        x,y,size = positions[i][0],positions[i][1],positions[i][2]
        input_stream = []
        # image read stream
        if i==0 :# init
            for j in range(POOL_NUM):
                img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+j-LEFT_FRAME,0,len(imagepaths)-1)])))
        else: # load next frame
            img_pool.pop(0)
            img_pool.append(impro.imread(os.path.join(opt.temp_dir+'/video2image',imagepaths[np.clip(i+LEFT_FRAME,0,len(imagepaths)-1)])))
        img_origin = img_pool[LEFT_FRAME]

        # preview result and print
        if not opt.no_preview:
            if show_pool.qsize()>3:   
                cv2.imshow('clean',show_pool.get())
                cv2.waitKey(1) & 0xFF

        if size>50:
            try:#Avoid unknown errors
                for pos in FRAME_POS:
                    input_stream.append(impro.resize(img_pool[pos][y-size:y+size,x-size:x+size], INPUT_SIZE,interpolation=cv2.INTER_CUBIC)[:,:,::-1])
                if init_flag:
                    init_flag = False
                    previous_frame = input_stream[N]
                    previous_frame = data.im2tensor(previous_frame,bgr2rgb=True,gpu_id=opt.gpu_id)
                
                input_stream = np.array(input_stream).reshape(1,T,INPUT_SIZE,INPUT_SIZE,3).transpose((0,4,1,2,3))
                input_stream = data.to_tensor(data.normalize(input_stream),gpu_id=opt.gpu_id)
                with torch.no_grad():
                    unmosaic_pred = netG(input_stream,previous_frame)
                img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = True)
                previous_frame = unmosaic_pred
                write_pool.put([False,imagepath,img_origin.copy(),img_fake.copy(),x,y,size])
            except Exception as e:
                init_flag = True
                print('Error:',e)
        else:
            write_pool.put([True,imagepath,img_origin.copy(),-1,-1,-1,-1])
            init_flag = True
        
        t2 = time.time()
        print('\r',str(i+1)+'/'+str(length),util.get_bar(100*i/length,num=35),util.counttime(t1,t2,i+1,len(imagepaths)),end='')
    print()
    write_pool.close()
    show_pool.close()
    if not opt.no_preview:
        cv2.destroyAllWindows()
    print('Step:4/4 -- Convert images to video')
    ffmpeg.image2video( fps,
                opt.temp_dir+'/replace_mosaic/output_%06d.'+opt.tempimage_type,
                opt.temp_dir+'/voice_tmp.mp3',
                 os.path.join(opt.result_dir,os.path.splitext(os.path.basename(path))[0]+'_clean.mp4')) 