import os
import random
from typing import List, Dict, Callable
from tests.e2e.e2e_base import E2EBase


class TestUntargetedAttack(E2EBase):
    """End-to-End test for untargeted attack with dynamic behavior"""
    COLLECT_INFO_PORT = 3431
    TEST_FOLDER = "untargeted_attack"

    @staticmethod
    def get_process_image_func() -> Callable[[bytes], Dict]:
        """Simplified dynamic image processing function using nonlocal counter"""
        animal_energies = {
            'zebra': 0.1,
            'horse': 0.8,
            'elephant': 0.15,
            'giraffe': 0.12,
            'deer': 0.2,
            'lion': 0.25,
            'dog': 0.18
        }

        trends = {
            'zebra': 0.001,
            'horse': -0.001,
            'elephant': 0.003,
            'giraffe': 0.0025,
            'deer': 0.0018,
            'lion': 0.0022,
            'dog': 0.0015
        }

        counter = None

        def process_image(encoded_image: bytes) -> Dict:
            nonlocal counter
            if counter is None:
                counter = 0
            else:
                counter += 1

            for animal in animal_energies:
                base_change = trends[animal]
                fluctuation = 0.0005 * random.random() * (counter % 3)
                animal_energies[animal] += base_change + fluctuation
                animal_energies[animal] = max(0.01, min(0.99, animal_energies[animal]))

            predictions = []
            for animal, score in animal_energies.items():
                if counter > 10 and random.random() < 0.2:
                    score *= 0.9

                predictions.append({
                    "label": animal,
                    "score": round(score, 4)
                })

            if counter % 7 == 0:
                predictions.sort(key=lambda x: x['label'])
            else:
                predictions.sort(key=lambda x: -x['score'])

            return {"predictions": predictions}

        return process_image


    @staticmethod
    def get_sys_argv() -> List[str]:
        """Dynamic configuration with some variability"""
        base_args = [
            "--attack_type", "untargeted",
            "--seed", "42",
            "--snapshot_interval", "1"
        ]

        return base_args

    def additional_assertions(self):
        """More flexible assertions with better diagnostics"""
        assert self._app_instance is not None, "App instance not initialized"

        output_folder = self._app_instance.config.get("output_dir")
        experiments_folder = self._app_instance.config.get("experiment_label")
        seed = self._app_instance.config.get("seed")

        save_folder = os.path.join(output_folder, experiments_folder, str(seed))
        self._verify_output_structure(save_folder)

        current_fitness = self._app_instance.stc.current_fitness_value
        assert current_fitness is not None, "Fitness value not calculated"
        assert current_fitness.value <= 0.35, f"Fitness {current_fitness.value} too high"

    def _verify_output_structure(self, save_folder: str):
        """Helper method to verify output structure with better error messages"""
        required_paths = {
            'base_folder': save_folder,
            'images_folder': os.path.join(save_folder, "images"),
            'stats_folder': os.path.join(save_folder, "statistics"),
            'final_image': os.path.join(save_folder, "images", "final_image.jpg"),
            'data_file': os.path.join(save_folder, "statistics", "data.json")
        }

        for name, path in required_paths.items():
            assert os.path.exists(path), f"Missing required path: {name} ({path})"
