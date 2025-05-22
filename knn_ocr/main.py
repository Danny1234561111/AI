import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def load_images_from_directory(dir_path: pathlib.Path) -> list:
    images = []
    for img_file in dir_path.glob("*.png"):
        img_data = convert_to_binary(plt.imread(str(img_file)))
        images.append(img_data)
    return images

def resize_image(image: np.array, target_size: list) -> np.array:
    height_diff = target_size[0] - image.shape[0]
    width_diff = target_size[1] - image.shape[1]
    return np.pad(image, ((height_diff // 2, height_diff // 2 + height_diff % 2),
                           (width_diff // 2, width_diff // 2 + width_diff % 2)))

def convert_to_binary(image: np.array) -> np.array:
    if len(image.shape) == 3:
        image = image.mean(axis=2)
    image[image > 0] = 1
    return image

def load_dataset_from_directory(dir_path: pathlib.Path) -> tuple:
    images = []
    labels = []
    symbol_mapping = {}
    max_dimensions = [0, 0]

    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            symbol = subdir.stem if len(subdir.stem) == 1 else subdir.stem[-1]
            label_index = len(symbol_mapping) + 1
            symbol_mapping[symbol] = label_index

            for img_file in subdir.glob("*.png"):
                img_data = convert_to_binary(plt.imread(img_file))
                images.append(img_data)
                labels.append(label_index)

                for dim in range(len(max_dimensions)):
                    if img_data.shape[dim] > max_dimensions[dim]:
                        max_dimensions[dim] = img_data.shape[dim]


    for idx, img in enumerate(images):
        images[idx] = resize_image(img, max_dimensions).flatten()

    return np.array(images), np.array(labels), symbol_mapping, max_dimensions

def extract_characters(image: np.array, max_dimensions: list) -> list:
    labeled_image = label(image)
    regions = regionprops(labeled_image)

    sorted_labels = {}
    for region in regions:
        sorted_labels[int(region.centroid[1])] = region.label

    sorted_labels = sorted(sorted_labels.items())
    for i, (x_pos, label_num) in enumerate(sorted_labels):
        if i > 0:
            distance = (x_pos - sorted_labels[i-1][0]) / image.shape[1]
            if distance < 0.02:
                labeled_image[labeled_image == label_num] = sorted_labels[i-1][1]

    character_images = {}
    boundaries = []
    for region in regionprops(labeled_image):
        boundaries.append((region.bbox[1], region.bbox[3]))
        character_images[int(region.centroid[1])] = convert_to_binary(resize_image(region.image, max_dimensions))

    boundaries = sorted(boundaries)
    for i, bound in enumerate(boundaries):
        if i > 0:
            distance = (bound[0] - boundaries[i-1][1]) / image.shape[1]
            if distance > 0.03:
                character_images[(bound[0] + boundaries[i-1][1]) / 2] = None

    return [img for _, img in sorted(character_images.items())]

# Main execution
base_path = pathlib.Path(__file__).parent
task_folder = base_path / "task"
training_folder = task_folder / "train"

text_images = load_images_from_directory(task_folder)
data_set, label_set, symbol_to_number, max_symbol_size = load_dataset_from_directory(training_folder)
number_to_symbol = {num: sym for sym, num in symbol_to_number.items()}

data_set = data_set.astype("float32")
label_set = label_set.reshape(-1, 1).astype("float32")

knn_model = cv2.ml.KNearest_create()
knn_model.train(data_set, cv2.ml.ROW_SAMPLE, label_set)

for idx, text_img in enumerate(text_images):
    extracted_symbols = extract_characters(text_img, max_symbol_size)
    result_string = ""
    for symbol in extracted_symbols:
        if symbol is None:
            result_string += " "
            continue
        flattened_symbol = symbol.flatten().reshape(1, -1).astype("float32")
        _, prediction, _, _ = knn_model.findNearest(flattened_symbol, 3)
        result_string += number_to_symbol[int(prediction)]

    print(f"{idx}) {result_string}")
