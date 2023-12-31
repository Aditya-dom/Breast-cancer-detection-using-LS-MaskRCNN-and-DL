import glob
import shutil

arr = []
dir_path = '/media/kazzastic/C08EBCFB8EBCEAD4/calc_training_full_roi_images/*png'

for filename in glob.glob(dir_path):
    arr.append(filename)


for i in range(len(arr)):
    if(arr[i].find('Calc-Training') != -1):
        print("Calc-Train Found")
        shutil.copy(arr[i], '/media/kazzastic/C08EBCFB8EBCEAD4/ROI_sorted/calc-train')
    elif(arr[i].find('Calc-Test') != -1):
        print("Calc-Test Found")
        shutil.copy(arr[i], '/media/kazzastic/C08EBCFB8EBCEAD4/ROI_sorted/calc-test')
    elif(arr[i].find('Mass-Test') != -1):
        print("Mass-Test Found")
        shutil.copy(arr[i], '/media/kazzastic/C08EBCFB8EBCEAD4/ROI_sorted/mass-test')
    elif(arr[i].find('Mass-Train') != -1):
        print("Mass-Train Found")
        shutil.copy(arr[i], '/media/kazzastic/C08EBCFB8EBCEAD4/ROI_sorted/mass-train')
    else:
        print("Nothing Found")