import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = np.zeros(590)  # TODO This supposes that the image height is always 720 BAD but it was fixed with the init function
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.zeros(3)  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = np.zeros(1)
        #distance in meters of vehicle center from the line
        self.line_base_pos = np.zeros(1)
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #smoothen the n frames 
        self.smoothen_nframes = 10
        #first frame 
        self.first_frame = True


left_lane = Line()
right_lane = Line()

# TODO this function should be called once for stream of images(video) and every time for separate images
def init_lines(width):
    global left_lane
    left_lane.bestx = np.zeros(width)
    # was the line detected in the last iteration?
    left_lane.detected = False
    # x values of the last n fits of the line
    left_lane.recent_xfitted = []
    # polynomial coefficients averaged over the last n iterations
    left_lane.best_fit = np.zeros(3)
    # polynomial coefficients for the most recent fit
    left_lane.current_fit = [np.array([False])]
    # radius of curvature of the line in some units
    left_lane.radius_of_curvature = np.zeros(1)
    # distance in meters of vehicle center from the line
    left_lane.line_base_pos = np.zeros(1)
    # difference in fit coefficients between last and new fits
    left_lane.diffs = np.array([0, 0, 0], dtype='float')
    # x values for detected line pixels
    left_lane.allx = None
    # y values for detected line pixels
    left_lane.ally = None
    # smoothen the n frames
    left_lane.smoothen_nframes = 10
    # first frame
    left_lane.first_frame = True

    global right_lane
    right_lane.bestx = np.zeros(width)
    # was the line detected in the last iteration?
    right_lane.detected = False
    # x values of the last n fits of the line
    right_lane.recent_xfitted = []
    # polynomial coefficients averaged over the last n iterations
    right_lane.best_fit = np.zeros(3)
    # polynomial coefficients for the most recent fit
    right_lane.current_fit = [np.array([False])]
    # radius of curvature of the line in some units
    right_lane.radius_of_curvature = np.zeros(1)
    # distance in meters of vehicle center from the line
    right_lane.line_base_pos = np.zeros(1)
    # difference in fit coefficients between last and new fits
    right_lane.diffs = np.array([0, 0, 0], dtype='float')
    # x values for detected line pixels
    right_lane.allx = None
    # y values for detected line pixels
    right_lane.ally = None
    # smoothen the n frames
    right_lane.smoothen_nframes = 10
    # first frame
    right_lane.first_frame = True
