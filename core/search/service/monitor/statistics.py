import os

import numpy as np
from PIL.Image import fromarray
from matplotlib import pyplot as plt

from core.search.service.archive import Archive
from core.search.service.monitor.search_listener import SearchListener
from core.search.service.search_time_controller import SearchTimeController
from core.search.solution import Solution


class Statistics(SearchListener):
    def __init__(self, stc: SearchTimeController, archive: Archive, config: dict):
        self.stc = stc
        self.archive = archive
        self.config = config
        self.snapshots = []
        self.snapshots_interval = self.config.get("snapshot_interval")
        self.snapshot_threshold = self.snapshots_interval
        self.stc.add_listener(self)
        seed = self.config.get("seed")
        experiment_label = self.config.get("experiment_label")

        output_dir = f"./{self.config.get("output_dir")}"
        experiment_folder = f"{output_dir}/{experiment_label}"
        exact_experiment_folder = f"{experiment_folder}/{seed}"
        statistics_folder = f"{exact_experiment_folder}/statistics"
        images_folder = f"{exact_experiment_folder}/images"

        self.directories = {
            "output_dir": output_dir,
            "experiment_folder": experiment_folder,
            "exact_experiment_folder": exact_experiment_folder,
            "statistics_folder": statistics_folder,
            "images_folder": images_folder
        }
        self.prepare_folders()

    def prepare_folders(self):
        for key in self.directories:
            if not os.path.exists(self.directories[key]):
                os.makedirs(self.directories[key])

    def new_action_evaluated(self):
        if self.snapshot_threshold < 0:
            return
        elapsed = self.stc.percentage_used_budget() * 100

        if elapsed >= self.snapshot_threshold:
            self.take_snapshot()

    def take_snapshot(self):
        solution = self.archive.extract_solution()
        self.snapshots.append(solution)
        self.snapshot_threshold += self.snapshots_interval

    def get_data(self, solution:Solution):
        # Get the data from the solution
        data = {
            'eval_count': self.stc.evaluated_individuals,
            'interval_count': self.snapshots_interval,
            'current_fitness': solution.fitness_value.value,
            'predictions': solution.fitness_value.predictions,
            'archive_size': solution.actions.__len__(),
            'percentage_used_budget': self.stc.percentage_used_budget(),
            'archive': self.archive.populations.copy()
        }
        return data

    def write_statistics(self):
        if self.snapshots_interval > 0:
            self.take_snapshot()
            self.write_snapshots()

        if self.config.get("write_statistics"):
            self.report_statistics()

        if self.config.get("save_images"):
            self.save_images()

    def report_statistics(self):
        pass

    def save_images(self):
        self.save_final_image()
        self.save_matrix_overlay()

    def write_snapshots(self):
        pass

    def save_final_image(self):
        img = self.archive.get_mutated_image()
        image = fromarray(img)
        image.save(f"{self.directories['images_folder']}/final_image.jpg")
        if self.config.get("show_plots"):
            plt.imshow(img)
            plt.show()

    def save_matrix_overlay(self):
        changes = self.archive.extract_solution()
        img = np.zeros([self.config.get("image_height"), self.config.get("image_width"), 3], dtype=np.uint8)
        img[:] = [255., 255., 255.]

        for point in changes.actions:
            img[point.get_location()[0]][point.get_location()[1]] = point.get_color()

        image = fromarray(img)
        image.save(f"{self.directories['images_folder']}/matrix_overlay.png")

        if self.config.get("show_plots"):
            plt.imshow(img)
            plt.show()