import ppo

if __name__ == "__main__":
    # default kernel is 2mm
    kernel_info = {
        "base_metrics": {
            "latency":    14687,
            "ff_usage":   14310,
            "lut_usage":  35715,
            "dsp_usage":  5,
            "bram_usage": 16,
        },

        "best_metrics": {
            "latency":    443,
            "ff_usage":   66911,
            "lut_usage":  48608,
            "dsp_usage":  412,
            "bram_usage": 38,
        },

        "loops": [
            {"name": "kernel/loop_1", "trip_count": 4,},
            {"name": "kernel/loop_2", "trip_count": 5,},
            {"name": "kernel/loop_3", "trip_count": 7,},
            {"name": "kernel/loop_4", "trip_count": 4,},
            {"name": "kernel/loop_5", "trip_count": 8,},
            {"name": "kernel/loop_6", "trip_count": 5,},
        ],

        "arrays": [
            {"name": "kernel tmp", "shape": (4, 5),},
            {"name": "kernel A",   "shape": (4, 7),},
            {"name": "kernel B",   "shape": (7, 5),},
            {"name": "kernel C",   "shape": (5, 8),},
            {"name": "kernel D",   "shape": (4, 8),},
        ],
    }

    ppo_agent_model_hidden_layer_dim = 256
    ppo_agent_model_path = "./ppo-agent-256.pth"

    ppo.train_ppo_agent(kernel_info, ppo_agent_model_hidden_layer_dim, ppo_agent_model_path)
    ppo.evaluate_ppo_agent(kernel_info, ppo_agent_model_hidden_layer_dim, ppo_agent_model_path)
    