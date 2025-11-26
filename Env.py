from typing import Optional
import copy
import subprocess
import xml.etree.ElementTree

import gymnasium as gym
import torch
import numpy as np
import sympy


class Env(gym.Env):
    def __init__(self, env_idx: int, designs_root_dir: str, job_scripts_root_dir: str, kernel_info: dict):
        self.env_idx: int = env_idx
        self.num_steps_taken: int = 0

        self.designs_root_dir: str = designs_root_dir
        self.job_scripts_root_dir: str = job_scripts_root_dir

        self.curr_metrics: dict = {
            "latency":    np.full(shape=(1,), fill_value=0, dtype=np.int64),
            "ff_usage":   np.full(shape=(1,), fill_value=0, dtype=np.int64),
            "lut_usage":  np.full(shape=(1,), fill_value=0, dtype=np.int64),
            "dsp_usage":  np.full(shape=(1,), fill_value=0, dtype=np.int64),
            "bram_usage": np.full(shape=(1,), fill_value=0, dtype=np.int64),
            # add more metrics here
        }

        # Metrics from the baseline design
        self.base_metrics: dict = {
            "latency":    np.full(shape=(1,), fill_value=kernel_info["base_metrics"]["latency"],    dtype=np.int64),
            "ff_usage":   np.full(shape=(1,), fill_value=kernel_info["base_metrics"]["ff_usage"],   dtype=np.int64),
            "lut_usage":  np.full(shape=(1,), fill_value=kernel_info["base_metrics"]["lut_usage"],  dtype=np.int64),
            "dsp_usage":  np.full(shape=(1,), fill_value=kernel_info["base_metrics"]["dsp_usage"],  dtype=np.int64),
            "bram_usage": np.full(shape=(1,), fill_value=kernel_info["base_metrics"]["bram_usage"], dtype=np.int64),
            # add more metrics here
        }

        # Metrics from the best design
        self.best_metrics: dict = {
            "latency":    np.full(shape=(1,), fill_value=kernel_info["best_metrics"]["latency"],    dtype=np.int64),
            "ff_usage":   np.full(shape=(1,), fill_value=kernel_info["best_metrics"]["ff_usage"],   dtype=np.int64),
            "lut_usage":  np.full(shape=(1,), fill_value=kernel_info["best_metrics"]["lut_usage"],  dtype=np.int64),
            "dsp_usage":  np.full(shape=(1,), fill_value=kernel_info["best_metrics"]["dsp_usage"],  dtype=np.int64),
            "bram_usage": np.full(shape=(1,), fill_value=kernel_info["best_metrics"]["bram_usage"], dtype=np.int64),
            # add more metrics here
        }

        self._set_legal_directives(kernel_info)

        self.action_space = gym.spaces.Discrete(len(self.legal_directives))

        self.observation_space = gym.spaces.Dict(
            spaces = {
                "latency":    gym.spaces.Box(low=1, high=2**63 - 2, shape=(1,), dtype=np.int64, seed=69),
                "ff_usage":   gym.spaces.Box(low=1, high=2**63 - 2, shape=(1,), dtype=np.int64, seed=69),
                "lut_usage":  gym.spaces.Box(low=1, high=2**63 - 2, shape=(1,), dtype=np.int64, seed=69),
                "dsp_usage":  gym.spaces.Box(low=1, high=2**63 - 2, shape=(1,), dtype=np.int64, seed=69),
                "bram_usage": gym.spaces.Box(low=1, high=2**63 - 2, shape=(1,), dtype=np.int64, seed=69),
                # add more observations here
            },
            seed = 88,
        )

        self.info: dict = {
            "legal_directive_idx_mask": np.full(shape=(len(self.legal_directives),), fill_value=1.0, dtype=np.float32),
            # add more info here
        }


    def _get_observation(self) -> dict:
        """
        Returns the observation of the environment.

        Returns:
            dict: The observation of the environment.
        """
        return self.curr_metrics


    def _get_info(self) -> dict:
        """
        Returns the information of the environment.

        Return:
            dict: The information of the environment.
        """
        return self.info


    def _set_legal_directives(self, kernel_info: dict):
        """
        Sets legal directives from kernel info.

        Args:
            kernel_info (dict): The kernel info from which legal directives are generated.
        """
        # each element denotes the starting index of the next segment (the first segment starts at index 0)
        directive_segments: list[int] = []

        curr_segment_directives_count: int = 0
    
        # "set_directive_pipeline loop_1 -II {1, 2, 4, 8}", "set_directive_pipeline loop_1 -off"
        legal_pipeline_directives = []
        for loop in kernel_info["loops"]:
            for i in [1, 2, 4, 8]:
                legal_pipeline_directives.append(f"set_directive_pipeline {loop['name']} -II {i}\n")
                curr_segment_directives_count += 1
            legal_pipeline_directives.append(f"set_directive_pipeline {loop['name']} -off\n")
            curr_segment_directives_count += 1
            directive_segments.append(curr_segment_directives_count)

        # "set_directive_unroll loop_1 -factor LOOP_TRIP_COUNT_DIVISOR"
        legal_unroll_directives = []
        for loop in kernel_info["loops"]:
            legal_unroll_factors = sympy.divisors(loop["trip_count"])
            for f in legal_unroll_factors:
                legal_unroll_directives.append(f"set_directive_unroll {loop['name']} -factor {f}\n")
                curr_segment_directives_count += 1
            directive_segments.append(curr_segment_directives_count)

        self.legal_directives = legal_pipeline_directives + legal_unroll_directives
        self.directive_segments = directive_segments


    def _add_directive(self, directive: str):
        """
        Adds one directive, if it does not exist, to `pragmas.tcl`.

        Args:
            directive (str): The directive to be added.
        """
        with open(file=f"{self.designs_root_dir}/design-{self.env_idx}/pragmas.tcl", mode="r+") as file:
            directives = file.read()
            if directive not in directives:
                file.write(directive)


    def _mask_directive_segment(self, directive_idx: int):
        """
        Masks a directive segment.

        Args:
            directive_idx (int): The index of the directive from the directive segment to be masked.
        """
        for i in range(len(self.directive_segments)):
            next_directive_segment_starting_idx = self.directive_segments[i]
            if directive_idx < next_directive_segment_starting_idx:
                if i == 0:
                    curr_directive_segment_starting_idx = 0
                    self.info["legal_directive_idx_mask"][curr_directive_segment_starting_idx:next_directive_segment_starting_idx - 1] = 0
                else:
                    curr_directive_segment_starting_idx = self.directive_segments[i - 1]
                    self.info["legal_directive_idx_mask"][curr_directive_segment_starting_idx:next_directive_segment_starting_idx - 1] = 0
            else:
                continue
        

    def _remove_all_directives(self):
        """
        Removes all directives from `pragmas.tcl`.
        """
        with open(file=f"{self.designs_root_dir}/design-{self.env_idx}/pragmas.tcl", mode="w+") as file:
            pass


    def _run_vitis(self):
        """
        Submits a Slurm job to run Vitis.
        """
        job_script_name = f"{self.job_scripts_root_dir}/run_vitis_{self.env_idx}.sh"
        subprocess.run(["sbatch", "--wait", "--quiet", job_script_name])


    def _update_metrics(self):
        """
        Updates the metrics from Vitis post-c-synthesis reports.
        """
        solution_path = f"{self.designs_root_dir}/design-{self.env_idx}/p1/s1"
        csynth_xml_path = f"{solution_path}/syn/report/csynth.xml"

        # Parse csynth.xml
        element_tree = xml.etree.ElementTree.parse(csynth_xml_path)
        root = element_tree.getroot()

        # Get the post C synthesis metrics
        if root.find(".//Average-caseLatency").text == "undef":
            latency = np.int64(root.find(".//Worst-caseLatency").text)
        else:
            latency = np.int64(root.find(".//Average-caseLatency").text)

        ff_usage   = np.int64(root.find(".//Resources/FF").text)
        lut_usage  = np.int64(root.find(".//Resources/LUT").text)
        dsp_usage  = np.int64(root.find(".//Resources/DSP").text)
        bram_usage = np.int64(root.find(".//Resources/BRAM_18K").text)

        # Update the metrics
        self.curr_metrics["latency"]    = np.full(shape=(1,), fill_value=latency,    dtype=np.int64)
        self.curr_metrics["ff_usage"]   = np.full(shape=(1,), fill_value=ff_usage,   dtype=np.int64)
        self.curr_metrics["lut_usage"]  = np.full(shape=(1,), fill_value=lut_usage,  dtype=np.int64)
        self.curr_metrics["dsp_usage"]  = np.full(shape=(1,), fill_value=dsp_usage,  dtype=np.int64)
        self.curr_metrics["bram_usage"] = np.full(shape=(1,), fill_value=bram_usage, dtype=np.int64)


    def _reset_metrics(self):
        """
        Resets the metrics to their baseline values.
        """
        self.curr_metrics: dict = copy.deepcopy(self.base_metrics)


    def _compute_reward_helper(self) -> float:
        """
        Computes reward using latency.

        Returns:
            float: The reward ranging from 0.0 to 1.0.
        """
        min_latency:  int = self.best_metrics["latency"].item()
        max_latency:  int = self.base_metrics["latency"].item()
        curr_latency: int = self.curr_metrics["latency"].item()

        max_reward:  int = -1 * min_latency
        min_reward:  int = -1 * max_latency
        curr_reward: int = -1 * curr_latency

        reward: float = (curr_reward - min_reward) / (max_reward - min_reward)

        return reward


    def _compute_reward(self) -> float:
        """
        Computes reward.

        Returns:
            float: The reward.
        """
        reward: float = self._compute_reward_helper()

        # Apply penalty for longer episode
        reward = reward - 0.001 * self.num_steps_taken

        return reward


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)

        self.num_steps_taken = 0

        self._remove_all_directives()
        self._reset_metrics()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info


    def step(self, action: torch.Tensor) -> tuple:
        terminated = False
        truncated = False

        self.num_steps_taken += 1

        directive_idx: int = action.item()

        directive: str = self.legal_directives[directive_idx]

        self._add_directive(directive)
        self._mask_directive_segment(directive_idx)
        self._run_vitis()
        self._update_metrics()

        reward: float = self._compute_reward()
        observation: dict = self._get_observation()
        info: dict = self._get_info()

        if self.curr_metrics["latency"].item() < self.best_metrics["latency"].item():
            terminated = True

        return observation, reward, terminated, truncated, info


gym.register(
    id = "yiyang/Env-v0",
    entry_point = Env,
    max_episode_steps = 64,
)
