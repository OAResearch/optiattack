# Configuration Parameters

## abc_limit

- **Default Value**: 10
- **Description**: Abandon limit for ABC algorithm. Number of trials before a food source is abandoned.

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

## attack_type

- **Default Value**: untargeted
- **Description**: Attack type for the search. Options: 'targeted' or 'untargeted'.

## base_endpoint

- **Default Value**: /api/v1
- **Description**: Base endpoint for the NUT. Default is '/api/v1'.

## baseline_attack

- **Default Value**: square
- **Description**: Baseline attack used by the 'baseline' algorithm. Options: 'square', 'hsj', 'simba', 'pixel' (ART), 'sparse_rs' (Sparse-RS L0, numpy port).

## baseline_stop_at_success

- **Default Value**: auto
- **Description**: Stop the baseline attack at the first successful query. 'auto' uses the attack default (square, simba, pixel: true; hsj: false, it uses the whole budget and reports the smallest successful perturbation). Options: 'auto', 'true', 'false'.

## crossover

- **Default Value**: single_point_crossover
- **Description**: Crossover operator for the search.

## cw_binary_search_steps

- **Default Value**: 10
- **Description**: Binary search steps of the C&W L2 attack over its constant (ART default).

## cw_confidence

- **Default Value**: 0.0
- **Description**: Confidence (logit margin) of the C&W L2 attack (ART default).

## cw_initial_const

- **Default Value**: 650.25
- **Description**: Initial constant of the C&W L2 attack on the 0-255 scale (ART default 0.01 on 0-1, times 255^2).

## cw_learning_rate

- **Default Value**: 0.01
- **Description**: Initial learning rate of the C&W L2 attack (ART default 0.01; chosen by measurement on the 0-255 scale, see README).

## cw_max_doubling

- **Default Value**: 5
- **Description**: Maximum doubling steps of the C&W L2 line search per iteration (ART default; each can cost a prediction query).

## cw_max_halving

- **Default Value**: 5
- **Description**: Maximum halving steps of the C&W L2 line search per iteration (ART default; each can cost a prediction query).

## cw_max_iter

- **Default Value**: 10
- **Description**: Iterations of the C&W L2 attack per binary search step (ART default). Each costs two gradient queries and up to cw_max_halving + cw_max_doubling + 1 prediction queries.

## enable_pruning

- **Default Value**: False
- **Description**: Enable pruning of the final results.

## enable_ui

- **Default Value**: False
- **Description**: Enable web interface.

## experiment_label

- **Default Value**: experiment
- **Description**: Experiment label.

## fgsm_eps

- **Default Value**: 8.0
- **Description**: FGSM (L-inf) step size on the 0-255 scale.

## final_image_name

- **Default Value**: final_image
- **Description**: Final image name.

## fitness_threshold

- **Default Value**: 0.0
- **Description**: Fitness threshold for stopping the search. Search stops when fitness <= this value. Default is 0.0 (stop when solution is found). Use negative values for stricter stopping criteria (e.g., -0.1).

## focused_search_activation_time

- **Default Value**: 0.8
- **Description**: The percentage of passed search before starting a more focused, less exploratory one

## hsj_init_eval

- **Default Value**: 100
- **Description**: Initial number of evaluations for gradient estimation in HopSkipJump (ART default).

## hsj_init_image

- **Default Value**: 
- **Description**: Path to an image used as the initial adversarial image of HopSkipJump (e.g. an image of the target class). Empty: random initialisation.

## hsj_init_size

- **Default Value**: 100
- **Description**: Maximum number of random trials for the initial adversarial image of HopSkipJump (ART default).

## hsj_max_eval

- **Default Value**: 10000
- **Description**: Maximum number of evaluations for gradient estimation in HopSkipJump (ART default).

## hsj_max_iter

- **Default Value**: 50
- **Description**: Maximum number of iterations of the HopSkipJump attack (ART default).

## hsj_norm

- **Default Value**: 2
- **Description**: Norm of the HopSkipJump attack. Options: '2' or 'inf'.

## image_height

- **Default Value**: 224
- **Description**: Image height in pixels. Should be same as the model input size.

## image_width

- **Default Value**: 224
- **Description**: Image width in pixels. Should be same as the model input size.

## input_image

- **Default Value**: ./tests/test_img.jpeg
- **Description**: Path to the input image.

## line_plot_name

- **Default Value**: line_plot
- **Description**: Line plot image name.

## max_action_size

- **Default Value**: 10
- **Description**: Maximum action size

## max_evaluations

- **Default Value**: 1000
- **Description**: Maximum number of evaluations for the search.

## max_pixels

- **Default Value**: 0
- **Description**: Maximum number of changed pixels (0: no limit). A pixel counts as changed when at least one channel differs from the original. Candidates that would exceed it are repaired before the NUT query by moving random actions on new pixels onto already-changed pixels; a candidate without actions is not queried. Works for all OptiAttack algorithms; the dense baselines (square, simba, hsj) reject it, and sparse_rs uses it as k.

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

## overlay_image_name

- **Default Value**: overlay_matrix
- **Description**: Overlay matrix image name.

## pgd_eps

- **Default Value**: 8.0
- **Description**: Maximum perturbation of the PGD attack in pgd_norm on the 0-255 scale (8/255 on the 0-1 scale for L-inf; use a larger value for L2).

## pgd_eps_step

- **Default Value**: 2.0
- **Description**: Step size of a PGD iteration in pgd_norm on the 0-255 scale.

## pgd_max_iter

- **Default Value**: 100
- **Description**: Number of PGD iterations; each costs one gradient query (ART default 100).

## pgd_norm

- **Default Value**: inf
- **Description**: Norm of the PGD attack. Options: 'inf' or '2'.

## pixel_es

- **Default Value**: 1
- **Description**: Optimiser of the PixelAttack. 0: CMA-ES, 1: differential evolution (ART default).

## pixel_max_iter

- **Default Value**: 100
- **Description**: Maximum number of optimiser iterations of the PixelAttack (ART default).

## pixel_th

- **Default Value**: 10
- **Description**: Number of pixels changed by the PixelAttack.

## population_size

- **Default Value**: 10
- **Description**: Population size of the population-based algorithms.

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

## shuffle_block_size

- **Default Value**: 3
- **Description**: Block size for shuffle mutator. Must be odd number (e.g., 3, 5, 7).

## simba_attack

- **Default Value**: px
- **Description**: SimBA basis. Options: 'px' or 'dct' (ART default; its steps are spread over the whole image and can vanish when rounded to uint8 on large images unless simba_epsilon is large).

## simba_epsilon

- **Default Value**: 25.5
- **Description**: SimBA step size on the 0-255 scale (ART default 0.1 is on 0-1).

## simba_freq_dim

- **Default Value**: 4
- **Description**: SimBA low frequency dimension for the 'dct' basis (ART default).

## simba_max_iter

- **Default Value**: 100000
- **Description**: Iteration limit of SimBA. High so that max_evaluations is the real limit.

## simba_order

- **Default Value**: random
- **Description**: SimBA order of the basis directions. Options: 'random' or 'diag' (ART default 'random').

## simba_stride

- **Default Value**: 1
- **Description**: SimBA stride for the 'dct' basis with 'diag' order (ART default).

## snapshot_interval

- **Default Value**: 5
- **Description**: Snapshot interval for the search.

## sparse_rs_k

- **Default Value**: 0
- **Description**: Number of pixels changed by Sparse-RS (L0). 0: use max_pixels, or 10 when max_pixels is 0 as well. Must not exceed max_pixels when that is set.

## sparse_rs_p_init

- **Default Value**: 0.3
- **Description**: Initial fraction of the k pixels that Sparse-RS moves per iteration; it shrinks with the official schedule. Default 0.3 is the official L0 default (eval.py, README for untargeted runs; README suggests 0.1 for targeted runs).

## square_eps

- **Default Value**: 8.0
- **Description**: Maximum perturbation of the Square attack on the 0-255 scale (ART default 0.3 is on 0-1).

## square_max_iter

- **Default Value**: 0
- **Description**: Iteration limit of the Square attack; its square size schedule is relative to it. 0 (auto): max_evaluations for the 'individual_evaluations' criterion, 10000 for 'time'.

## square_nb_restarts

- **Default Value**: 1
- **Description**: Number of restarts of the Square attack (ART default).

## square_norm

- **Default Value**: inf
- **Description**: Norm of the Square attack. Options: 'inf' or '2'.

## square_p_init

- **Default Value**: 0.8
- **Description**: Initial fraction of the image changed by a Square attack update (ART default).

## statistics_column_id

- **Default Value**: -
- **Description**: Statistics column id.

## statistics_file_name

- **Default Value**: statistics
- **Description**: Statistics file name.

## stopping_criterion

- **Default Value**: individual_evaluations
- **Description**: Stopping criterion for the search. Options: 'individual_evaluations' or 'time'.

## target

- **Default Value**: None
- **Description**: Target class for the targeted attack. If not specified, any misclassification is considered successful.

## use_gradients

- **Default Value**: False
- **Description**: Allow OptiAttack algorithms to use gradients of a gradient-capable NUT (GradientOracle). Must stay false for black-box results; no algorithm uses it yet. Needs a NUT that declares 'gradients'.

## whitebox_attack

- **Default Value**: pgd
- **Description**: White-box attack used by the 'whitebox' algorithm (ART, needs a NUT with 'gradients' and 'logits'). Options: 'fgsm', 'pgd', 'cw' (Carlini-Wagner L2). Gradient queries count towards max_evaluations.

## write_statistics

- **Default Value**: True
- **Description**: Write the statistics to a file.

## zero_mutation_rate

- **Default Value**: 0.4
- **Description**: Mutation rate for the zero mutation.

