import cv2

def click_on_point(img):
    # create a window and display the image

    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # define lists to store the clicked points
    x = []
    y = []

    # define a function to handle mouse clicks
    def handle_click(event, x_click, y_click, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            x.append(x_click)
            y.append(y_click)
            print('Clicked at:', x_click, y_click)

    # set the mouse callback to handle clicks
    cv2.setMouseCallback('image', handle_click)

    # run the event loop and handle events
    while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:  # exit if 'q' or 'esc' key is pressed
            break

    # return the clicked points
    if len(x)>0  and len(y)>0:
        return [(x[i], y[i]) for i in range(len(x))]
    else:
        return [None]
