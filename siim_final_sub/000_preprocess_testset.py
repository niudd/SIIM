import glob
import pydicom
import cv2


#raw_test_path = 'data/dicom-images-test/*/*/*.dcm'
raw_test_path = 'data/dicom-images-test/*.dcm'
test_fname_list = glob.glob(raw_test_path)
print(len(test_fname_list))

## convert all dcm images into png images ##
# test folder
for idx, raw_fname in enumerate(test_fname_list):
    dataset = pydicom.dcmread(raw_fname)
    fname = raw_fname.split('/')[-1][:-4]
    new_fname = 'data/test/%s.png'%fname
    cv2.imwrite(new_fname, dataset.pixel_array)
    #if idx>10:
    #    break

print('Complete')