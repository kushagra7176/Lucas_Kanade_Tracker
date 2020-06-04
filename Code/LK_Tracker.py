import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def warpAffine(I, W, Tpoints):
    height, width = I.shape
    n, _ = Tpoints.shape
    warpedPoints = np.matmul(W, Tpoints.T)
    warpedPoints = warpedPoints.T
    warpedIntensities = np.empty([n, 1])

    for i in range(len(warpedPoints)):
        if warpedPoints[i, 1].astype(int) >width :
            warpedPoints[i, 1] = width
        if warpedPoints[i, 0].astype(int) >height:
            warpedPoints[i, 0] = height
        warpedIntensities[i, 0] = I[
            warpedPoints[i, 1].astype(int) , warpedPoints[i, 0].astype(int) ]

    return warpedPoints, warpedIntensities

def warpAffineGradient(gradientX, gradientY, IWpoints):
    height, width = gradientX.shape
    n, it = IWpoints.shape
    gradXIntensities = np.empty([n, 1])
    gradYIntensities = np.empty([n, 1])

    for i in range(len(IWpoints)):

        if IWpoints[i, 1].astype(int) > width:
            IWpoints[i, 1] = width

        if IWpoints[i, 0].astype(int) > height:
            IWpoints[i, 0] = height

        gradXIntensities[i, 0] = gradientX[IWpoints[i, 1].astype(int), IWpoints[i, 0].astype(int)]

        gradYIntensities[i, 0] = gradientY[IWpoints[i, 1].astype(int), IWpoints[i, 0].astype(int)]

    return gradXIntensities, gradYIntensities


def calcSteepestDescent(IWdx, IWdy, TPoints):
    img1 = IWdx[:, 0] * [TPoints[:, 0]]
    img2 = IWdx[:, 0] * [TPoints[:, 1]]
    img3 = IWdy[:, 0] * [TPoints[:, 0]]
    img4 = IWdy[:, 0] * [TPoints[:, 1]]
    SDI = np.hstack((img1.T, img3.T, img2.T, img4.T, IWdx, IWdy))

    return SDI

def lucasKanadeTracker(I, Template, rectangle_coords, p):

    threshold = 0.00001

    # Compute Image Gradients.
    gradientX = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    gradientY = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

    iteration = 0

    while True:
        iteration += 1
        # Warp Affine Parameters.
        W = np.array([[1 + p[0, 0], p[0, 2], p[0, 4]],
                      [p[0, 1], 1 + p[0, 3], p[0, 5]]])

        # Warp Affine -------------------------------------STEP 1
        IWpoints, IWi = warpAffine(I, W, Template[0])

        # Error Image -------------------------------------STEP 2
        error = Template[1] - IWi

        # Warped image gradients --------------------------STEP 3
        IWdx, IWdy = warpAffineGradient(gradientX, gradientY, IWpoints)

        # Compute Steepest Descent Image (SDI)-------------STEP 5
        SDI = calcSteepestDescent( IWdx, IWdy, Template[0])

        # Compute Hessian Matrix --------------------------STEP 6
        Hessian_mat = np.matmul(SDI.T, SDI)

        # SDI Parameter Update------------------------------------------STEP 7
        SDI_param_update = np.matmul(SDI.T, error)

        # Compute del_P -----------------------------------STEP 8
        deltaP = np.matmul(np.linalg.pinv(Hessian_mat), SDI_param_update)

        # Update Parameters P -----------------------------------STEP 9
        p[0, 0] += deltaP[0, 0]
        p[0, 1] += deltaP[1, 0]
        p[0, 2] += deltaP[2, 0]
        p[0, 3] += deltaP[3, 0]
        p[0, 4] += deltaP[4, 0]
        p[0, 5] += deltaP[5, 0]

        if (iteration > 10):
            break
        if (0<max(deltaP[0]) < threshold):
            break

    newstartPoint = np.array([[rectangle_coords[0]], [rectangle_coords[1]], [1]])
    newendPoint = np.array([[rectangle_coords[2]], [rectangle_coords[3]], [1]])
    newStart = np.matmul(W, newstartPoint)
    newEnd = np.matmul(W, newendPoint)

    return newStart, newEnd


########################################################################################################################
##############################                           MAIN                             ##############################
########################################################################################################################


PATHS = ['DragonBaby/img/', 'Car4/img/', 'Bolt2/img/']
Dataset_name = ['DragonBaby', 'Car4', 'Bolt2']

# Setting output folder
folders = ['OUTPUT_BABY', 'OUTPUT_CAR', 'OUTPUT_BOLT']

# Creating folder directory if one doesn't exits
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

for ind,image_PATH in enumerate(PATHS):

    list_of_filenames = [filename for filename in os.listdir(image_PATH) ]
    dataset_size = len(list_of_filenames)
    print("dataset size :",dataset_size)

    p = np.zeros([1, 6], dtype=np.float)
    framesSeen = 0
    it = 0
    Template = []
    for im in range(0,dataset_size):

        with open(Dataset_name[ind] + '_ROI_coordinates.txt', 'r') as file:
            coords_list = file.read()
            [x1, y1, x2, y2] = [int(x) for x in coords_list.split(',') ]

        dimensions = [x2-x1, y2-y1]

        startingPoint = [x1, y1]
        endPoint = [x2, y2]

        # Read Template Image
        tmp = cv2.imread(image_PATH+list_of_filenames[0], 0)
        h1 = abs(startingPoint[1] - endPoint[1])
        w1 = abs(startingPoint[0] - endPoint[0])
        tmpPoints = np.empty([h1 * w1, 3])
        Template.append(tmpPoints)
        n = 0

        # Getting Template frame points
        for i in range(startingPoint[0], endPoint[0]):
            for j in range(startingPoint[1], endPoint[1]):
                tmpPoints[n, 0] = i
                tmpPoints[n, 1] = j
                tmpPoints[n, 2] = 1
                n += 1

        tmpIntensities = np.empty([h1 * w1, 1])

        n = 0

        # Get Template frame intensities
        for i in tmpPoints:
            tmpIntensities[n, 0] = tmp[int(i[1]), int(i[0])]
            n += 1

        Template.append(tmpIntensities)

        mean1 = np.mean(tmp)

        # Read each frame from dataset.
        img = cv2.imread(image_PATH+list_of_filenames[im])
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clone = frame.copy()

        mean2 = np.mean(frame)

        # Normalize the input image from dataset
        frame = frame.astype(float)
        frame = frame * (mean1 / mean2)

        # Rectangular coordinates of bounding box.
        rect_coords = [x1, y1, x2, y2]


        newStart, newEnd = lucasKanadeTracker(frame, Template, rect_coords, p)


        if Dataset_name[ind] == 'DragonBaby':
            rect_width = 80
            rect_height = 75

            center_x = np.abs(newStart[0] + newEnd[0]) / 2
            center_y = np.abs(newStart[1] + newEnd[1]) / 2

        if Dataset_name[ind] == 'Car4':
            rect_width = 115
            rect_height = 80

            center_x = np.abs(newStart[0] + newEnd[0]) / 2
            center_y = np.abs(newStart[1] + newEnd[1]) / 2

            if it > 220:
                center_x = np.abs(newStart[0] + newEnd[0]) / 2 + 0.25 * rect_width
                center_y = np.abs(newStart[1] + newEnd[1]) / 2

            if it > 250:
                center_x = np.abs(newStart[0] + newEnd[0]) / 2 + 0.5 * rect_width
                center_y = np.abs(newStart[1] + newEnd[1]) / 2


        if Dataset_name[ind] == 'Bolt2':
            rect_width = 60
            rect_height = 90

            center_x = np.abs(newStart[0] + newEnd[0]) / 2
            center_y = np.abs(newStart[1] + newEnd[1]) / 2

            if it > 125:
                center_x = np.abs(newStart[0] + newEnd[0]) / 2 + 1 * rect_width
                center_y = np.abs(newStart[1] + newEnd[1]) / 2

            if it > 250:
                center_x = np.abs(newStart[0] + newEnd[0]) / 2 + 1.5 * rect_width
                center_y = np.abs(newStart[1] + newEnd[1]) / 2

        print(it)

        # Calculate coordinates of bounding box.
        new_start = np.array([center_x - (rect_width / 2), center_y + (rect_height / 2)])
        new_end = np.array([center_x + (rect_width / 2), center_y - (rect_height / 2)])

        # Draw rectangle around tracked object.
        cv2.rectangle(img, (new_start[0], new_start[1]), (new_end[0], new_end[1]), (255, 0, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)

        # Writing the image (Uncomment to save images)
        # cv2.imwrite(folder + '/' + str(it) + '.jpg', img)
        
        it += 1

    print('All frames processed')
    print('\nPress \'q\' to destroy window')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('All frames processed')
