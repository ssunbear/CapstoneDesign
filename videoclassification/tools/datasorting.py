import os
from PIL import Image

def allfiles(path):
    res = []
    
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)

        for file in files:
            filepath = os.path.join(rootpath, file)
            res.append(filepath)

    return res

if __name__ == "__main__":
    strSrc = "/mnt3/nas/1-56/AIVIS/1인칭/"
    #testTxtFiles = ["/datasets/carAccident_videoClassification_1/test_1st.txt", "/datasets/carAccident_videoClassification_1/test_2nd.txt", "/datasets/carAccident_videoClassification_1/test_3rd.txt", "/datasets/carAccident_videoClassification_1/test_4th.txt"]
    testTxtFiles = ["/home/kmkwak/DeepLearning/AIVIS/datasets/carAccident_videoClassification_1/test_1st_kor.txt"]
    #strSrc = "/datasets/carAccident_videoClassification_1/"
    #imageFolderPath = strSrc + "image_jpg/"  ############ modify after view1 sending!!!!!!!!

    for i in range(0, len(testTxtFiles)):
        textlines = open(testTxtFiles[i], 'r')
        lines = textlines.readlines()

        for j in range(0, len(lines)):
            print("%d test, %d / %d lines" % (i, j, len(lines)))

            checkTxt = lines[j].split('\n')[0].split(' ')

            imageFolderPath = strSrc + checkTxt[0]
            newFolderPath = "/home/kmkwak/DeepLearning/AIVIS/datasets/carAccident_videoClassification_1/" + checkTxt[0]
            newFolderPath = newFolderPath.replace('/image_jpg/', '/image_jpg_new/')

            file_list = allfiles(imageFolderPath)

            file_list_jpg = [file for file in file_list if file.endswith('.jpg')]
            print("Jpg File Length : " + str(len(file_list_jpg)))

            for k in range(0, len(file_list_jpg)):
                jpgFilePath = file_list_jpg[k].split("/")[-1]
                imageFilePath = newFolderPath + jpgFilePath


                if os.path.isfile(imageFilePath):
                    print("%d / %d\n" % (i, len(file_list_jpg)))
                    continue

                if (i%1000 == 0) :
                    print(str(i) + ".open " + jpgFilePath)

                img = Image.open(jpgFilePath)
                if img.mode == "RGBA" :
                    img.load()
                    newImg = Image.new("RGB", img.size, (255, 255, 255))
                    newImg.paste(img, mask = img.split()[3])
                    img = newImg

                if not os.path.exists(imageFilePath[:len(imageFilePath) - len(imageFilePath.split("/")[-1])]):
                    os.makedirs(imageFilePath[:len(imageFilePath) - len(imageFilePath.split("/")[-1])])
                try:
                    img.save(imageFilePath, "JPEG", quality=100)
                except:
                    print("%s failed to save\n" % (imageFilePath))
          

    
    
    # file_list = allfiles(imageFolderPath)

    # file_list_jpg = [file for file in file_list if file.endswith('.jpg')]
    # print("Jpg File Length : " + str(len(file_list_jpg)))

    # f = open("/datasets/carAccident_videoClassification/convertlog.txt", 'w')

    # for i in range(0, len(file_list_jpg)):
    #     ########################################################################
    #     ############ Load File
    #     jpgFilePath = file_list_jpg[i]
    #     imageFilePath = jpgFilePath.replace("/image_jpg/", "/image_jpg_new/") 
    #     #jsonFileName = jsonFilePath.split("/")[-1]
    #     #imageFileName = imageFilePath.split("/")[-1]
    #     #save_jsonFilePath = strDst + "/" + jsonFileName
    #     #save_imageFilePath = strDst + "/" + imageFileName

    #     if os.path.isfile(imageFilePath):
    #         print("%d / %d\n" % (i, len(file_list_jpg)))
    #         continue

    #     if (i%1000 == 0) :
    #         print(str(i) + ".open " + jpgFilePath)
            
    #     #if not os.path.isfile(imageFilePath):
    #     #    continue
        
    #     img = Image.open(jpgFilePath)
    #     if img.mode == "RGBA" :
    #         img.load()
    #         newImg = Image.new("RGB", img.size, (255, 255, 255))
    #         newImg.paste(img, mask = img.split()[3])
    #         img = newImg

    #     if not os.path.exists(imageFilePath[:len(imageFilePath) - len(imageFilePath.split("/")[-1])]):
    #         os.makedirs(imageFilePath[:len(imageFilePath) - len(imageFilePath.split("/")[-1])])
    #     try:
    #         img.save(imageFilePath, "JPEG", quality=100)
    #     except:
    #         f.write("%s failed to save\n" % (imageFilePath))

    # f.close()

