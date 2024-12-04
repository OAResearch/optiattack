from io import BytesIO

import numpy as np
from PIL import Image

from core.remote.remote_controller import RemoteController

controller = RemoteController()

controller.run_nut()
image_data = BytesIO()
image = Image.new("RGB", (224, 224), color="red")
image.save(image_data, format="JPEG")
image_data.seek(0)

image_array = np.array(image)

for i in range(100):
    res = controller.new_action(image_array)
    print(f"Action {i} completed")

print("All actions completed")
controller.stop_nut()


