"""Class to handle the statistics of the search process."""
import csv
import json
import os

import numpy as np
from PIL.Image import fromarray
from matplotlib import pyplot as plt

from core.search.service.archive import Archive
from core.search.service.monitor.search_listener import SearchListener
from core.search.service.search_time_controller import SearchTimeController
from core.search.solution import Solution


class Statistics(SearchListener):

    """Class to handle the statistics of the search process."""

    def __init__(self, stc: SearchTimeController, archive: Archive, config: dict):
        """Initialize the statistics."""

        self.stc = stc
        self.archive = archive
        self.config = config
        self.snapshots: list[Solution] = []
        self.snapshots_interval = self.config.get("snapshot_interval")
        self.snapshot_threshold = self.snapshots_interval
        self.stc.add_listener(self)

        self.output_dir = f"{self.config.get('output_dir')}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.final_image_name = self.config.get("final_image_name")
        self.overlay_image_name = self.config.get("overlay_image_name")
        self.line_plot_name = self.config.get("line_plot_name")
        self.statistics_file_name = self.config.get("statistics_file_name")

    def new_action_evaluated(self):
        """Check if a snapshot should be taken."""
        if self.snapshot_threshold < 0:
            return
        elapsed = self.stc.percentage_used_budget() * 100

        if elapsed >= self.snapshot_threshold:
            self.take_snapshot()

    def take_snapshot(self):
        """Take a snapshot of the current state of the search."""
        solution = self.archive.extract_solution()
        self.snapshots.append(solution)
        self.snapshot_threshold += self.snapshots_interval

    def get_data(self, solution: Solution):
        """Get the data from the solution."""
        data = {
            'eval_count': self.stc.get_evaluated_individuals(),
            'interval_count': self.snapshots_interval,
            'current_fitness': solution.fitness_value.value,
            'predictions': solution.fitness_value.predictions,
            'archive_size': solution.actions.__len__(),
            'percentage_used_budget': self.stc.percentage_used_budget(),
            'archive': self.archive.populations.copy()
        }
        return data

    def write_statistics(self):
        """Write the statistics to a file."""
        if self.snapshots_interval > 0:
            self.take_snapshot()
            self.write_snapshots()

        if self.config.get("write_statistics"):
            self.report_statistics()

        if self.config.get("save_images"):
            self.save_images()

    def report_statistics(self):
        """Report the statistics of the search."""
        self.save_statistics()

    def save_images(self):
        """Save generated images. It contains the final image and the matrix overlay."""
        self.save_final_image()
        self.save_matrix_overlay()
        self.save_line_plot()

    def write_snapshots(self):
        """Write the snapshots to a file."""
        pass

    def save_final_image(self):
        """Save the final image. It contains the actions of the archive."""
        img = self.archive.get_mutated_image()
        image = fromarray(img)
        image.save(f"{self.output_dir}/{self.final_image_name}.jpg")
        if self.config.get("show_plots"):
            plt.imshow(img)
            plt.show()
        return image

    def save_matrix_overlay(self):
        """Save the matrix overlay. It contains the actions of the archive."""
        changes = self.archive.extract_solution()
        img = np.zeros([self.config.get("image_height"), self.config.get("image_width"), 3], dtype=np.uint8)
        img[:] = [255., 255., 255.]

        for point in changes.actions:
            img[point.get_location()[0]][point.get_location()[1]] = point.get_color()

        image = fromarray(img)
        image.save(f"{self.output_dir}/{self.overlay_image_name}.png")

        if self.config.get("show_plots"):
            plt.imshow(img)
            plt.show()

    def save_line_plot(self):
        """Save the line plot. It contains the predictions of the search."""
        plt.clf()
        legend = []
        data = {}
        data_counter = 0
        for val in self.snapshots:
            for s in val.fitness_value.predictions:
                if s.label not in data:
                    # if not exists on the beginning add zeros
                    data[s.label] = [0] * data_counter
                data[s.label].append(s.value)

        for item in data:
            plt.plot(np.arange(len(data[item])), data[item], label=item)
            legend.append(item)

        plt.xlabel('Generations')
        plt.ylabel('Prediction Fitness Value')
        ax = plt.subplot(111)
        chart_box = ax.get_position()
        ax.set_position([chart_box.x0, chart_box.y0, chart_box.width * 0.8, chart_box.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)
        plt.savefig(f"{self.output_dir}/{self.line_plot_name}.png", bbox_inches='tight')

        if self.config.get("show_plots"):
            plt.show()

    def save_statistics(self):
        """Save the data as a json file."""
        data = {}
        data['execution_time'] = self.stc.get_elapsed_seconds()
        data['eval_count'] = self.stc.get_evaluated_individuals()
        data['interval_count'] = self.snapshots_interval
        data['current_fitness'] = self.stc.current_fitness_value.value

        if data['current_fitness'] <= 0:
            data['flipped'] = True
        else:
            data['flipped'] = False

        predictions = {}
        data_counter = 0
        for val in self.snapshots:
            for s in val.fitness_value.predictions:
                if s.label not in predictions:
                    # if not exists on the beginning add zeros
                    predictions[s.label] = [0] * data_counter
                predictions[s.label].append(s.value)

        data['predictions'] = predictions
        data['action_size'] = len(self.archive.extract_solution().actions)
        matrix = self.archive.extract_solution().actions.copy()
        changes = []
        for point in matrix:
            changes.append({
                'location': point.get_location(),
                'color': point.get_color()
            })

        data['changes'] = changes

        if self.config.get("enable_pruning"):
            not_minimized_matrix = self.archive.extract_solution(not_minimized=True).actions.copy()
            not_minimized_changes = []
            for point in not_minimized_matrix:
                not_minimized_changes.append({
                    'location': point.get_location(),
                    'color': point.get_color()
                })

            data['not_minimized__size'] = len(not_minimized_changes)
            data['not_minimized_changes'] = not_minimized_changes
        else:
            data['not_minimized_size'] = None
            data['not_minimized_changes'] = []

        data['config'] = self.config

        with open(f"{self.output_dir}/{self.statistics_file_name}.json", 'w') as outfile:
            json.dump(data, outfile, indent=6, sort_keys=True, default=str)

        filtered_data = {}

        for k, v in data.items():
            if isinstance(v, (dict, list)):
                if k == "config" and isinstance(v, dict):
                    for ck, cv in v.items():
                        filtered_data[f"config_{ck}"] = cv
            else:
                filtered_data[k] = v

        with open(f"{self.output_dir}/{self.statistics_file_name}.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=filtered_data.keys())
            writer.writeheader()
            writer.writerow(filtered_data)
