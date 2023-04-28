import cv2
import PySimpleGUI as sg


def relabel_or_continue():
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [
        [sg.Text('Please write the label name below ->')],
        [sg.Text('Label name:'), sg.InputText()],
        [sg.Button('CONTINUE'), sg.Button('CANCEL'), sg.Button('END')]
    ]

    # Create the Window
    window = sg.Window('Continue labeling with another label:', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            event = 'CANCEL'
        print(f'You entered {values[0]}')
        break

    window.close()
    return values[0], event

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

if __name__ == "__main__":
    relabel_or_continue()