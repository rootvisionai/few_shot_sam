import copy
import json

import cv2
import PySimpleGUI as sg
import interface_utils as utils
import glob, os


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

def negative_embedding():
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = []

    # Create the Window
    window = sg.Window('Add negative points ...', layout)
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

    cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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


    annotations = []
    encoded_images = []
    image_paths = []
    for image_id, image_path in enumerate(glob.glob(os.path.join("..", "support_images", "*.tiff"))):
        init_image = utils.import_image(image_path)
        image_shape = init_image.shape
        window_size = (512, 512)
        image = cv2.resize(copy.deepcopy(init_image), (window_size[0], window_size[1]))
        while True:
            print("CLICK ON POSITIVE POINTS")
            points = click_on_point(img=image)
            positive_points = []
            if not None in points:
                for pt in points:
                    xy = utils.adapt_point(
                        {"x": pt[0], "y": pt[1]},
                        initial_shape=image.shape[0:2],
                        final_shape=image_shape[0:2]
                    )
                    positive_points.append([xy["x"], xy["y"]])

            print("CLICK ON NEGATIVE POINTS")
            points = click_on_point(img=image)
            negative_points = []
            if not None in points:
                for pt in points:
                    xy = utils.adapt_point(
                        {"x": pt[0], "y": pt[1]},
                        initial_shape=image.shape[0:2],
                        final_shape=image_shape[0:2]
                    )
                    negative_points.append([xy["x"], xy["y"]])

            label, event = relabel_or_continue()

            annotations.append({
                "coordinates": {"positive": positive_points, "negative": negative_points},
                "label": label,
                "image_id": image_id,
                "image_path": image_path
            })

            if event == "END":
                break
            elif event == "CANCEL":
                break
            else:
                pass

        encoded_images.append(utils.numpy_to_base64(init_image))

    json_file = {
        "annotations": annotations,
        "images": encoded_images,
    }

    with open("request_content.json", "w") as fp:
        json.dump(json_file, fp, indent=3)

    print(f"ANNOTATIONS:\n{annotations}")



