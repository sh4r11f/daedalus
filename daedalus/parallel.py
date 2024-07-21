#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: parallel.py
#
#
#                   DESCRIPTION: Classes and function for parallel processing
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 07-19-2024-7810598105114117
#                         SPACE: Dartmouth College, Hanover, NH
#
# =================================================================================================== #
from pathlib import Path


class ScriptGenerator:
    """
    A class to generate SLURM scripts.

    """
    def __init__(self, version, params, exec_dir=None, cluster_name="Discovery"):

        self.version = version
        self.params = params
        self.exec_dir = exec_dir
        self.cluster_name = cluster_name
        # ----------------------------------------------------------------------------------------------------------- #
        # Settings
        self.shorthand = self.params["shorthand"]
        self.array_ids = self.params["array_ids"]
        # ----------------------------------------------------------------------------------------------------------- #
        # Directories and files
        cluster_settings = self.params["Platform"][self.cluster_name]
        self.modules = cluster_settings["modules"]
        self.root = Path(cluster_settings["Directories"]["root"])
        self.log_dir = self.root / "log" / self.cluster_name / f"v{self.version}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python_x = [mod for mod in self.modules if mod["name"] == "python"][0]["path"]
        self.conda_x = [mod for mod in self.modules if mod["name"] == "conda"][0]["path"]
        # ----------------------------------------------------------------------------------------------------------- #
        # SLURM settings
        resources = cluster_settings["resources"]
        self.account = resources["account"]
        self.partition = resources["partition"]
        self.n_nodes = int(resources["nodes"])
        self.max_ntasks = int(resources["tasks"])
        self.n_cpus = int(resources["cpus_per_task"])
        self.mem_gb = int(resources["memory_per_cpu"])
        self.wall_time = resources["wall_time"]

    def parallel_script(self, job_name, srun_cmds, conda_env, exec_dir=None, job_time=None):
        """

        Args:
            job_name (str): The name of the job.
            srun_cmds (list): srun commands to run in parallel.
            exec_dir (str or Path): The directory to execute the script in. Defaults to None.

        Returns:
            dict: The SLURM scripts to test and run the job.
        """
        # Setup
        exec_dir = self.exec_dir if exec_dir is None else exec_dir
        wall_time = self.wall_time if job_time is None else job_time
        n_tasks = len(srun_cmds)
        if n_tasks > self.max_ntasks:
            print(f"^^INFO^^ JOB {job_name}: Number of requested tasks ({n_tasks}) exceeds the maximum per node.")
            print(f"^^INFO^^ JOB {job_name}: Splitting the job into multiple nodes.")
            n_nodes = (n_tasks // self.max_ntasks) + 1
        else:
            n_nodes = self.n_nodes
        log_dir = self.log_dir / f"{job_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        # ----------------------------------------------------------------------------------------------------------- #
        # Write the SLURM script
        script = "#!/bin/bash -l\n\n"

        # Basics
        script += f"#SBATCH --job-name={job_name[:4]}\n"
        script += f"#SBATCH --account={self.account}\n"
        script += f"#SBATCH --partition={self.partition}\n"
        if exec_dir is not None:
            script += f"#SBATCH --chdir={exec_dir}\n"
        script += f"#SBATCH --time={wall_time}\n"
        script += f"#SBATCH --nodes={n_nodes}\n"
        script += f"#SBATCH --ntasks={n_tasks}\n"
        script += f"#SBATCH --cpus-per-task={self.n_cpus}\n"
        script += f"#SBATCH --mem-per-cpu={self.mem_gb}G\n"
        script += f"#SBATCH --output={str(log_dir)}/job-%A_sub-%a.out\n"
        script += f"#SBATCH --array={','.join(self.array_ids)}\n\n"

        # Add modules
        for module in self.modules:
            script += f"module load {module}\n"
            if module == "conda":
                script += f"source {self.conda_x}\n"
            script += f"conda activate {conda_env}\n\n"

        # Add variables
        ids = " ".join(self.array_ids)
        script += f"declare -a do_ids=( {ids} )\n"
        script += f"SLURM_DOID=${{do_ids[$SLURM_ARRAY_TASK_ID - 1]}}\n"
        script += f"log_file={str(log_dir)}/job-${{SLURM_ARRAY_JOB_ID}}_sub-${{SLURM_DOID}}_${{SLURM_ARRAY_TASK_ID}}\n\n"

        # Print out some information
        script += "date\n"
        script += f'echo "THIS IS JUST THE BEGINNING OF {job_name} FOR ${{SLURM_DOID}}..."\n\n'

        # Add the srun commands
        for n, cmd in enumerate(srun_cmds):
            srun_cmd  = f"srun --nodes=1 --ntasks=1 --cpus-per-task=${{SLURM_CPUS_PER_TASK}} \\\n"
            srun_cmd += f"{cmd} \\\n"
            srun_cmd += f"1> ${{log_file}}_{n:02d}.out \\\n"
            srun_cmd += f"2> ${{log_file}}_{n:02d}.err &\n"
            srun_cmd +=  "# ---------------------------------------------------------------------------------- #\n"
            script += srun_cmd
        script += "wait\n\n"

        # Print out more information
        script += f'echo "THIS IS THE END OF {job_name} FOR ${{SLURM_DOID}}."\n'
        script += "date\n"

        return script

    def test_script(self, **config):
        """
        Make a slurm script with some config to test

        Args:
            config (dict): The configuration to test

        Returns:
            str: The SLURM script
        """
        wall = config.get("wall", self.wall)
        account = config.get("account", self.account)
        partition = config.get("partition", self.partition)
        n_nodes = config.get("nodes", self.n_nodes)
        n_tasks = config.get("tasks", 1)
        n_cpus = config.get("cpus_per_task", self.n_cpus)
        mem_gb = config.get("memory_per_cpu", self.mem_gb)

        # Write the SLURM script
        script = "#!/bin/bash -l\n\n"
        # Basics
        script += "#SBATCH --job-name=test\n"
        script += f"#SBATCH --account={account}\n"
        script += f"#SBATCH --partition={partition}\n"
        script += f"#SBATCH --time={wall}\n"
        script += f"#SBATCH --nodes={n_nodes}\n"
        script += f"#SBATCH --ntasks={n_tasks}\n"
        script += f"#SBATCH --cpus-per-task={n_cpus}\n"
        script += f"#SBATCH --mem-per-cpu={mem_gb}G\n"
        script += "#SBATCH --output=test.out\n"
        script += "#SBATCH --error=test.err\n\n"

        return script
