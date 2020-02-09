import numpy as np
import argparse
import cv2


def define_argParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--puzzle", required=True,
                    help="Path to the puzzle image")
    ap.add_argument("-w", "--query", required=True,
                    help="Path to the waldo image")
    return ap


def get_argParser():
    ap = define_argParser()
    return vars(ap.parse_args())


args = get_argParser()

# open puzzle and waldo image
puzzle = cv2.imread(args["puzzle"])  # args["puzzle"]
waldo = cv2.imread(args["query"])
(waldoHeight, waldoWidth) = waldo.shape[:2]

# find wald using correlation coefficient method
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCORR_NORMED)
# get the max and min values. MT_CCORR best match has highest value
# (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result) # fastest way to get the best matched coordinates, but not to get multiple ones
# display the match map
matchResult = cv2.resize(result, (int(result.shape[1] * 0.2), int(result.shape[0] * 0.2)), interpolation=cv2.INTER_AREA)
cv2.imshow("match result", matchResult)

# sort the results of matchResult, to get multiple hits
reshapedResults = np.reshape(result, result.shape[0] * result.shape[1])
sortedResults = np.argsort(reshapedResults)

# save the coordinates of the 3 best matched locations
topResultsArray = []
for i in range(0, 3):
    topResultsArray.append(np.unravel_index(sortedResults[-i], result.shape))

# copy out the best ROIs best matched to the query image and save them to copy them back into the puzzle after the mask
rois = []
for (y, x) in topResultsArray:
    botRight = (x + waldoWidth, y + waldoHeight)
    roi = puzzle[y:botRight[1], x:botRight[0]]
    rois.append(roi)

# mask the puzzle. Waldos need to be cut out before masking and applied after
mask = np.zeros(puzzle.shape, dtype="uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

# add query image back into the puzzle
i = 0
for (y, x) in topResultsArray:
    botRight = (x + waldoWidth, y + waldoHeight)
    roi = puzzle[y:botRight[1], x:botRight[0]]
    puzzle[y:botRight[1], x:botRight[0]] = rois[i]
    i = i + 1

# display images
puzzle = cv2.resize(puzzle, (int(puzzle.shape[1] * 0.35), int(puzzle.shape[0] * 0.35)), interpolation=cv2.INTER_AREA)
cv2.imshow("Puzzle", puzzle)
cv2.imshow("waldo", waldo)
cv2.waitKey(0)
