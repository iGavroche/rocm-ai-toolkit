import os
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from toolkit.paths import CONFIG_ROOT

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf('device', 'cpu')
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        print("[DEBUG] ExtensionJob.run() - calling super().run()")
        import sys
        sys.stdout.flush()
        super().run()

        print("[DEBUG] ExtensionJob.run() - super() completed")
        sys.stdout.flush()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")
        sys.stdout.flush()

        for i, process in enumerate(self.process):
            print(f"[DEBUG] ExtensionJob.run() - starting process {i}: {type(process).__name__}")
            sys.stdout.flush()
            process.run()
            print(f"[DEBUG] ExtensionJob.run() - process {i} completed")
            sys.stdout.flush()
