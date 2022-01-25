import itertools


def main():
    model_name = "segnet"
    pruning_ratios = [0, 0.5, 0.75, 0.875, 0.9375]
    pruning_durations = [0, 50, 100, 200, 300, 400, 450]

    seeds = list(range(2, 6))

    var_configs = itertools.product(pruning_ratios, pruning_durations)
    values = (
        (sparsity, duration, True, 1, f"{model_name}_{sparsity}_{duration}") for sparsity, duration in var_configs
    )

    annealing = (
        (
            sparsity,
            300,
            False,
            seed,
            f"{model_name}_{sparsity}_300_seed{seed}_annealed",
        )
        for sparsity, seed in itertools.product(pruning_ratios, seeds)
    )

    annealing_reference = (
        (sparsity, 300, True, seed, f"{model_name}_{sparsity}_300_seed{seed}")
        for sparsity, seed in itertools.product(pruning_ratios, seeds)
    )

    base_config = "camvid_segnet_base.yaml"

    with open(base_config, "r") as stream:
        config_str = stream.read()

    commands = []
    for sparsity, duration, learnable, seed, name in itertools.chain(values, annealing, annealing_reference):
        new_config_str = config_str
        new_config_str = new_config_str.replace("MODEL", name)
        new_config_str = new_config_str.replace("SPARSITY", str(sparsity))
        new_config_str = new_config_str.replace("DURATION", str(duration))
        new_config_str = new_config_str.replace("LEARNABLE", str(learnable))
        new_config_str = new_config_str.replace("SEED", str(seed))
        new_config_str = new_config_str.replace("SPARSE", "true" if sparsity > 0 else "false")

        new_config_fp = base_config.replace("segnet_base", name)
        with open(new_config_fp, "w") as file:
            file.write(new_config_str)

        if "anneal" in name:
            commands.append(f"bash tool/train.sh camvid {name}")

        bash_command_helper = "bash_commands.txt"
        buckets = [[] for _ in range(9)]

        for i, cmd in enumerate(commands):
            buckets[i % len(buckets)].append(cmd)

        command_strs = [" && ".join(bucket) + "\n" for bucket in buckets]

        with open(bash_command_helper, "w") as file:
            file.writelines(command_strs)


if __name__ == "__main__":
    main()
