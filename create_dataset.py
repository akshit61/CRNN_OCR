import os
import glob
import pickle as p
import config

def create_pickle():
    for set in os.listdir(config.DATA_ROOT):
        list_img_path = []
        list_labels = []
        images = glob.glob(config.DATA_ROOT+'/'+set+'/*.jpg')
        for image in images:
            img_name = image.split('\\')[-1][:-4]
            list_img_path.append(img_name)
            with open(os.path.join(config.DATA_ROOT,set,img_name+'.txt'),'r') as label_file:
                label = label_file.read()
                list_labels.append(label)
        pk = dict(zip(list_img_path,list_labels))
        with open(os.path.join(config.DATA_ROOT,set[5:]+'.pickle'), 'wb') as handle:
            p.dump(pk, handle, protocol=p.HIGHEST_PROTOCOL)

if __name__== '__main__':
    create_pickle()