import os
import sys
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import trange, tqdm
from typing import Dict, List, Any, Optional, Tuple, Union


BLUE = "\x1b[34m"
RESET = "\x1b[0m"
RED = "\x1b[31m"


def convertImagesToBin(input_dir: str, output_dir: str, crop_size : Optional[str]="224", resize_size : Optional[str]="224") -> None:
    """
    This function converts the test Dataset, pointed by input_dir, into a new folder (output_dir / crop_size) that will contain .bin files,  useful for the 
    inference on the Fusion target.


    Parameters
    ----------
    - input_dir: str
    The path where the Dataset is present. 
    - output_dir: str
    The output where the file will be saved. 
    - crop_size: str
    The final image size.
    - resize_size: str 
    The resize size. 

    
    Returns
    -------
    - None

    """
    source_path = Path(input_dir) / "test"
    size_value = crop_size
    resize_size = int(resize_size)
    output_path = Path(output_dir) / size_value

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    if output_path.exists():
        shutil.rmtree(output_path)

    os.mkdir(output_path)

    t = trange(100, desc="["+ BLUE + "INFO" + RESET + "]" +" CONVERTING ", bar_format='{desc}: {n_fmt}/{total_fmt} files [{elapsed}]')

    for i, img_file in enumerate(source_path.rglob("*")):
        if img_file.suffix.lower() in extensions:
            t.set_description("["+ BLUE + "INFO" + RESET + "]" + " CONVERTING (file %i)" % i, refresh=True)
            t.update()

            relative_path = img_file.relative_to(source_path)

            dest_file = output_path / relative_path.with_suffix(".bin")

            dest_file.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(img_file) as img:

                w, h = img.size

                if w < h:
                    new_w = resize_size
                    new_h = int(h * (resize_size / w))
                else:
                    new_h = resize_size
                    new_w = int(w * (resize_size / h))

                img_resized = img.resize((new_w, new_h), Image.BILINEAR)

                left = (new_w - int(crop_size)) / 2
                top = (new_h - int(crop_size)) / 2
                right = (new_w + int(crop_size)) / 2
                bottom = (new_h + int(crop_size)) / 2
                img_cropped = img_resized.crop((left, top, right, bottom))

                img_array= np.array(img_cropped, dtype=np.uint8)

                img_array.tofile(dest_file)


def createClassesFile(input_dir: str, output_dir: str) -> None:
    """
    This function creates the classes.h file that is useful to perform the inferencing on
    Fusion Platform.
    The .h file will contain two different #define: CLASSES and an CLASS_TABLE (an X-Define). 

    Parameters
    ----------
    - input_dir: str
    The path where the Dataset is present. 

    - output_dir: str
    The output where the file will be saved. 

    Returns
    -------
    - None

    """


    source_path = Path(input_dir) / "test"
    classes_h_file_path = Path(output_dir) / "classes.h"
    classes = sorted(entry.name for entry in os.scandir(source_path) if entry.is_dir())

    try:

        if classes_h_file_path.exists():
            os.remove(classes_h_file_path)

        with open(classes_h_file_path, 'a') as f:
            f.write(f"#define CLASSES {len(classes)}\n")
            f.write(f"#define CLASSES_TABLE")
            for idx, class_name in enumerate(classes):
                f.write(f"   X({idx}, {class_name}, \"{class_name}\")")


    except OSError as e:
        print(f"["+ RED + "ERROR" + RESET + "]" + f"Encountered a problem opening the file.\nThe specific error is: {e}")
    except Exception as e:
        print(f"["+ RED + "ERROR" + RESET + "]" + f"Encountered a generic problem creating the h classes file.\nThe specific error is: {e}")

    

#The Input is give by the dataset path
convertImagesToBin(sys.argv[1], "Converters/FusionConverter/DatasetConverter/DatasetFiles", sys.argv[2], sys.argv[3])
createClassesFile(sys.argv[1], "Converters/FusionConverter/DatasetConverter/DatasetFiles")