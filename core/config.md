# Configuration Parameters

## algorithm

- **Default Value**: mio
- **Description**: Search algorithm for the optimization.

## apc_location_end

- **Default Value**: 30
- **Description**: The end value of the adaptive parameter control for the location

## apc_location_start

- **Default Value**: 40
- **Description**: The start value of the adaptive parameter control for the location

## apc_pixel_end

- **Default Value**: 30
- **Description**: The end value of the adaptive parameter control for the pixel value

## apc_pixel_start

- **Default Value**: 40
- **Description**: The start value of the adaptive parameter control for the pixel value

## apc_start_time

- **Default Value**: 0.4
- **Description**: The start value of the adaptive parameter control

## apc_threshold

- **Default Value**: 0.6
- **Description**: The threshold value of the adaptive parameter control

## base_endpoint

- **Default Value**: /api/v1
- **Description**: Base endpoint for the NUT. Default is '/api/v1'.

## enable_pruning

- **Default Value**: False
- **Description**: Enable pruning of the final results.

## enable_ui

- **Default Value**: False
- **Description**: Enable web interface.

## experiment_label

- **Default Value**: experiment
- **Description**: Experiment label.

## focused_search_activation_time

- **Default Value**: 0.8
- **Description**: The percentage of passed search before starting a more focused, less exploratory one

## image_height

- **Default Value**: 224
- **Description**: Image height in pixels. Should be same as the model input size.

## image_width

- **Default Value**: 224
- **Description**: Image width in pixels. Should be same as the model input size.

## input_image

- **Default Value**: ./tests/test_img.jpeg
- **Description**: Path to the input image.

## max_action_size

- **Default Value**: 10
- **Description**: Maximum action size

## max_evaluations

- **Default Value**: 1000
- **Description**: Maximum number of evaluations for the search.

## min_action_size

- **Default Value**: 1
- **Description**: Minimum action size

## mutation_sigma

- **Default Value**: 50
- **Description**: Sigma value for the gaussian noise.

## mutator

- **Default Value**: one_zero_mutator
- **Description**: Mutation operator for the search.

## nut_host

- **Default Value**: localhost
- **Description**: Host address for the NUT. Default is 'localhost'.

## nut_port

- **Default Value**: 38000
- **Description**: Port number for the NUT. Default is 38000.

## one_mutation_rate

- **Default Value**: 0.3
- **Description**: Mutation rate for the one mutation.

## output_dir

- **Default Value**: ./output
- **Description**: Path to the output directory.

## pruning_method

- **Default Value**: standard
- **Description**: Pruning method for the search.

## random_sampling_probability

- **Default Value**: 0.5
- **Description**: Probability of sampling a new individual at random

## sampler

- **Default Value**: random_sampler
- **Description**: Sampler type for the search.

## save_images

- **Default Value**: True
- **Description**: Save the generated images.

## seed

- **Default Value**: -1
- **Description**: Seed number for the random number generator. Negative values mean use the system time.

## show_plots

- **Default Value**: True
- **Description**: Show plots.

## show_progress

- **Default Value**: True
- **Description**: Show progress of the search.

## snapshot_interval

- **Default Value**: 5
- **Description**: Snapshot interval for the search.

## stopping_criterion

- **Default Value**: individual_evaluations
- **Description**: Stopping criterion for the search. Options: 'individual_evaluations' or 'time'.

## target

- **Default Value**: None
- **Description**: Target class for the targeted attack. If not specified, any misclassification is considered successful.

## write_statistics

- **Default Value**: True
- **Description**: Write the statistics to a file.

## zero_mutation_rate

- **Default Value**: 0.4
- **Description**: Mutation rate for the zero mutation.

