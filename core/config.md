# Configuration Parameters

## image_height

- **Default Value**: 224
- **Description**: Image height in pixels. Should be same as the model input size.

## image_width

- **Default Value**: 224
- **Description**: Image width in pixels. Should be same as the model input size.

## input_image

- **Default Value**: ../tests/test_img.jpeg
- **Description**: Path to the input image.

## max_evaluations

- **Default Value**: 100
- **Description**: Maximum number of evaluations for the search.

## nut_host

- **Default Value**: localhost
- **Description**: Host address for the NUT. Default is 'localhost'.

## nut_port

- **Default Value**: 38000
- **Description**: Port number for the NUT. Default is 38000.

## seed

- **Default Value**: -1
- **Description**: Seed number for the random number generator. Negative values mean use the system time.

## show_progress

- **Default Value**: True
- **Description**: Show progress of the search.

## stopping_criterion

- **Default Value**: individual_evaluations
- **Description**: Stopping criterion for the search. Options: 'action_evaluations', 'individual_evaluations', 'time'.

