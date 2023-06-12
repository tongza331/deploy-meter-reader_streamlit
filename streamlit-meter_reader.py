import streamlit as st 
import numpy as np
import cv2
from model_utils import *
from PIL import Image

st.header("Industrial Meter Reader 🤖") 
st.write("To study how we can read meter value from image using deep learning.") 

SEG_LABEL = {'background': 0, 'pointer': 1, 'scale': 2}

file = st.uploaded_file = st.file_uploader("Upload your meter image here...", type=['png','jpeg','jpg'])
if file is not None:
    image = file.read()
    # img = st.image(image, caption='Original image', use_column_width=True)
    original_image = Image.open(file)
    image = np.array(original_image)
    st.write(":green[READY TO READ!]")

if st.button('READ METER'):
    im_shape = np.array([[input_shape, input_shape]]).astype('float32')
    scale_factor = np.array([[1, 2]]).astype('float32')
    input_image = det_preprocess(image, input_shape)
    inputs_dict = {'image': input_image, "im_shape": im_shape, "scale_factor": scale_factor}

    # Run meter detection model
    det_results = detector.predict(inputs_dict)

    # Filter out the bounding box with low confidence
    filtered_results = filter_bboxes(det_results, score_threshold)

    # Prepare the input data for meter segmentation model
    scale_x = image.shape[1] / input_shape * 2
    scale_y = image.shape[0] / input_shape

    # Create the individual picture for each detected meter
    roi_imgs, loc = roi_crop(image, filtered_results, scale_x, scale_y)
    roi_imgs, resize_imgs = roi_process(roi_imgs, METER_SHAPE)

    # Create the pictures of detection results
    roi_stack = np.hstack(resize_imgs)

    ### Run meter segmentation model
    seg_results = list()
    mask_list = list()
    num_imgs = len(roi_imgs)

    # Run meter segmentation model on all detected meters
    for i in range(0, num_imgs, seg_batch_size):
        batch = roi_imgs[i : min(num_imgs, i + seg_batch_size)]
        seg_result = segmenter.predict({"image": np.array(batch)})
        seg_results.extend(seg_result)

    results = []
    for i in range(len(seg_results)):
        results.append(np.argmax(seg_results[i], axis=0)) 
    seg_results = erode(results, erode_kernel)

    # # Create the pictures of segmentation results
    for i in range(len(seg_results)):
        mask_list.append(segmentation_map_to_image(seg_results[i], COLORMAP))
    mask_stack = np.hstack(mask_list)

    rectangle_meters = circle_to_rectangle(seg_results)
    line_scales, line_pointers = rectangle_to_line(rectangle_meters)
    binaried_scales = mean_binarization(line_scales)
    binaried_pointers = mean_binarization(line_pointers)
    scale_locations = locate_scale(binaried_scales)
    pointer_locations = locate_pointer(binaried_pointers)
    pointed_scales = get_relative_location(scale_locations, pointer_locations)
    meter_readings = calculate_reading(pointed_scales)

    rectangle_list = list()
    # Plot the rectangle meters
    for i in range(len(rectangle_meters)):
        rectangle_list.append(segmentation_map_to_image(rectangle_meters[i], COLORMAP))
    rectangle_meters_stack = np.hstack(rectangle_list)

    for i in range(len(meter_readings)):
        st.write("Meter {}: {:.3f}".format(i + 1, meter_readings[i]))
    
    result_image = image.copy()
    for i in range(len(loc)):
        cv2.rectangle(result_image,(loc[i][0], loc[i][1]), (loc[i][2], loc[i][3]), (0, 150, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(result_image, (loc[i][0], loc[i][1]), (loc[i][0] + 100, loc[i][1] + 40), (0, 150, 0), -1)
        cv2.putText(result_image, f"Meter{i+1}: {meter_readings[i]:.3f}", (loc[i][0],loc[i][1] + 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    st.image(result_image)


## Reference
st.write("Reference: https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/203-meter-reader/203-meter-reader.ipynb")