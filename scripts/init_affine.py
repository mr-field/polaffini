import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import numpy as np
import SimpleITK as sitk      
from dwarp import utils
import polaffini.polaffini as polaffini
import argparse

parser = argparse.ArgumentParser(description="POLAFFINI segmentation-based initialization for non-linear registration to template.")

# inputs
parser.add_argument('-m', '--mov-img', type=str, required=True, help='Path to the moving images (can use *).')
parser.add_argument('-ms', '--mov-seg', type=str, required=True, help='Path to the moving segmentations (can use *, should have same alphabetical order as the images).')
parser.add_argument('-ma', '--mov-aux', type=str, required=False, default=None, help='Path to the moving auxiliary images (can use *, should have same alphabetical order as the images).')
parser.add_argument('-r', '--ref-img', type=str, required=False, default=None, help="Path to the reference template image, can be 'mni1' or 'mni2")
parser.add_argument('-rs', '--ref-seg', type=str, required=False, default=None, help="Path to the reference template segmentation, can be 'mni1' or 'mni2")
parser.add_argument('-ra', '--ref-aux', type=str, required=False, default=None, help='Path to the reference template auxiliary image.')
# outputs
parser.add_argument('-o', '--out-dir', type=str, required=True, help='Path to output directory.')
parser.add_argument('-os', '--out-seg', type=int, required=False, default=0, help='Also output moved segmentations (1:yes, 0:no). Default: 0.')
parser.add_argument('-oa', '--out-aux', type=int, required=False, default=0, help='Also output moved auxiliary images (1:yes, 0:no). Default: 0.')
parser.add_argument('-ohot', '--one-hot', type=int, required=False, default=1, help='Perform one-hot encoding on moved output segmentations (1:yes, 0:no). Default: 1.')
parser.add_argument('-kpad', '--k-padding', type=int, required=False, default=5, help='Pad an image such that image size along each dimension becomes of form 2^k (k must be greater than the number of contracting levels). Default: 5.')

args = parser.parse_args()
args.out_seg = bool(args.out_seg)
args.one_hot = bool(args.one_hot)

#%% 

def init_polaffini_set(mov_files, 
                       mov_seg_files,
                       ref_seg,
                       labs,
                       resampler,
                       out_dir,
                       mov_aux_files=None,
                       out_seg=False,
                       one_hot=True):

    for i in range(len(mov_files)):
        mov_seg = sitk.ReadImage(mov_seg_files[i])
        
        # rescale intensities, one hot encoding, write images  
        resampler.SetReferenceImage(ref_seg)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetInterpolator(sitk.sitkLinear)
        mov_img = sitk.Cast(sitk.ReadImage(mov_files[i]), sitk.sitkFloat32)
        mov_img = resampler.Execute(mov_img)
        mov_img = utils.normalize_intensities(mov_img)
        sitk.WriteImage(mov_img, os.path.join(out_dir,'img',os.path.split(mov_files[i])[-1]))
        if mov_aux_files is not None:
            mov_aux = sitk.Cast(sitk.ReadImage(mov_aux_files[i]), sitk.sitkFloat32)
            mov_aux = resampler.Execute(mov_aux)
            mov_aux = utils.normalize_intensities(mov_aux, wmax=1)
            sitk.WriteImage(mov_aux, os.path.join(out_dir,'auxi',os.path.split(mov_aux_files[i])[-1]))
        if out_seg:
            resampler.SetOutputPixelType(sitk.sitkInt16)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mov_seg = resampler.Execute(mov_seg)
            if one_hot:
                mov_seg = utils.one_hot_enc(mov_seg, labs)
            sitk.WriteImage(mov_seg, os.path.join(out_dir,'seg',os.path.split(mov_seg_files[i])[-1]))

#%% Main

if args.ref_seg == "mni2" or args.ref_img == "mni2":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
    args.ref_img = os.path.join(maindir, 'refs', 'mni_brain_2mm.nii.gz')
elif args.ref_seg == "mni1" or args.ref_img == "mni1":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')
    args.ref_img = os.path.join(maindir, 'refs', 'mni_brain.nii.gz')  
    
os.makedirs(os.path.join(args.out_dir), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'img'), exist_ok=True)
if args.out_seg:
    os.makedirs(os.path.join(args.out_dir, 'seg'), exist_ok=True)
if args.mov_aux:
    os.makedirs(os.path.join(args.out_dir, 'auxi'), exist_ok=True)
   
mov_files = sorted(glob.glob(args.mov_img))
mov_seg_files = sorted(glob.glob(args.mov_seg))
if args.mov_aux is not None:
    mov_aux_files = sorted(glob.glob(args.mov_aux))
else:
    mov_aux_files = None
    
ref = sitk.Cast(sitk.ReadImage(args.ref_img), sitk.sitkInt16)
ref = utils.normalize_intensities(ref)
ref = utils.pad_image(ref, k=args.k_padding)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ref)   

resampler.SetInterpolator(sitk.sitkNearestNeighbor)  
resampler.SetOutputPixelType(sitk.sitkFloat32)
ref_seg = sitk.ReadImage(args.ref_seg)
ref_seg = resampler.Execute(ref_seg)
labs = np.unique(sitk.GetArrayFromImage(ref_seg))
labs = np.delete(labs, labs==0)
   
if args.ref_aux is not None:
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetInterpolator(sitk.sitkLinear)
    ref_aux = sitk.ReadImage(args.ref_aux)
    ref_aux = resampler.Execute(ref_aux)
    ref_aux = utils.normalize_intensities(ref_aux,wmax=1)
      
inshape = ref.GetSize()
ndims = ref.GetDimension()

print('\nPOLAFFINI initialization on set (n='+str(len(mov_files))+'):')
init_polaffini_set(mov_files=mov_files, 
                    mov_seg_files=mov_seg_files,
                    mov_aux_files=mov_aux_files,
                    ref_seg=ref_seg,
                    labs=labs,
                    resampler=resampler,
                    out_dir=args.out_dir,
                    out_seg=args.out_seg,
                    one_hot=args.one_hot)

sitk.WriteImage(ref, os.path.join(args.out_dir, 'ref_img.nii.gz'))
if args.ref_aux:
    sitk.WriteImage(ref_aux, os.path.join(args.out_dir, 'ref_aux.nii.gz'))
if args.out_seg:  
    if args.one_hot:
        ref_seg = utils.one_hot_enc(ref_seg, labs)
    sitk.WriteImage(ref_seg, os.path.join(args.out_dir, 'ref_seg.nii.gz'))
