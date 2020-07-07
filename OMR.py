import cv2
import numpy as np
import utils


#=========================PARAMETERS================================
path = "Images/mcq1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,1,2,4]
#===================================================================


img = cv2.imread(path)

# ===========Preprocessing==================
img = cv2.resize(img,(widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),3)
imgCanny = cv2.Canny(imgBlur,10,50)


# ===========Finding all Contours==================
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(255,0,255),7)


# =================Find Rectangles===================
rectCont = utils.rectContour(contours)

biggestContour = utils.getCornerPoints(rectCont[0]) # First Biggest area
gradePoints = utils.getCornerPoints(rectCont[1])    # second biggest for grading

if len(biggestContour)!=0 and len(gradePoints!=0):
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,0,255),15)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),15)

    # Reorder points for our bird view
    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)
    

    # ====================== Bird View for Biggest Rectangle (OMR)============================
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # ============= Bird View for Second Biggest Rectangle (Grading) ============================
    ptsG1 = np.float32(gradePoints)  
    ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  
    matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) 
    #cv2.imshow("grade",imgGradeDisplay)


    # ======================== Apply Threshold ==================================
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV)[1]


    # =================== Split our omr sheet into 25 images(this case) ===================
    boxes = utils.splitBoxes(imgThresh,questions,choices)
    #cv2.imshow("split",boxes[0])


    # ====================== Store non zero pixels for each 25 images(this case) =======================
    countR=0
    countC=0
    myPixelVal = np.zeros((questions,choices)) 
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]= totalPixels
        countC += 1
        if (countC == choices):
            countC = 0
            countR += 1

    #print(myPixelVal)

    # ================ Store index of max value of non zero pixels (answers) for each question in a list =================
    myIndex = []
    for x in myPixelVal:
        myIndex.append(np.where(x == np.amax(x))[0][0])
    #print(myIndex)


    # ====================== Grading =============================
    grading = []
    for x in range(questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    score = (sum(grading)/questions)*100
    #cv2.putText(imgGradeDisplay,str(int(score)) + " % ", (50,50), cv2.FONT_ITALIC, 1.5, (0,255,0),2)
    #cv2.imshow("grade",imgGradeDisplay)


    # =============================== Show Answers ===========================
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
    
    # create new blank image like warped image and put its inverse perspective on original image
    imgRawDrawings = np.zeros_like(imgWarpColored) 
    imgRawDrawings = utils.showAnswers(imgRawDrawings,myIndex,grading,ans,questions,choices)

    # ============== Inverse Perspective of Biggest rectangle(OMR)==========================
    invmatrix = cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invmatrix, (widthImg, heightImg))

    # ============== Inverse Perspective of Seccond Biggest rectangle(Grade)==========================
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade,str(int(score)) + "%", (50,100), cv2.FONT_ITALIC, 3, (0,0,255), 5)
    #cv2.imshow("grade",imgRawGrade)

    invmatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invmatrixG, (widthImg, heightImg))
    #cv2.imshow("Inv grade",imgInvGradeDisplay)

    imgFinal = cv2.addWeighted(imgFinal,0.75,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal,0.75,imgInvGradeDisplay,1,0)
    #cv2.imshow("Final Img", imgFinal)



lables = [  ["Original", "Gray", "Blur", "Canny"],
            ["Contours", "Biggest Contour", "Warpped", "Threshold"],
            ["Result", "Raw Drawing", "Inverse Warp", "Final"]  
        ]

blank_img = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],          
            [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
            [imgResult, imgRawDrawings, imgInvWarp, imgFinal])

imageStacked = utils.stackImages(imgArray, 0.32)


cv2.imshow("Stacked Img", imageStacked)
cv2.waitKey(0)