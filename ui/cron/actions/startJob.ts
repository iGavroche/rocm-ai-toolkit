import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken } from '../paths';
const isWindows = process.platform === 'win32';

const startAndWatchJob = (job: Job) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async (resolve, reject) => {
    const jobID = job.id;

    // setup the training
    const trainingRoot = await getTrainingFolder();

    const trainingFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    // make the config file
    const configPath = path.join(trainingFolder, '.job_config.json');

    //log to path
    const logPath = path.join(trainingFolder, 'log.txt');

    try {
      // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
      // if the log path does not exist, create it
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        let num = 0;
        while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
          num++;
        }

        fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
      }
    } catch (e) {
      console.error('Error moving log file:', e);
    }

    // update the config dataset path
    const jobConfig = JSON.parse(job.job_config);
    jobConfig.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

    // write the config file
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    let pythonPath = 'python';
    // use .venv or venv if it exists
    if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
      if (isWindows) {
        pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe');
      } else {
        pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
      }
    } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
      if (isWindows) {
        pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe');
      } else {
        pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
      }
    }

    const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
    if (!fs.existsSync(runFilePath)) {
      console.error(`run.py not found at path: ${runFilePath}`);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: run.py not found`,
        },
      });
      return;
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      // ROCm/AMD GPU support - HIP_VISIBLE_DEVICES is the ROCm equivalent of CUDA_VISIBLE_DEVICES
      HIP_VISIBLE_DEVICES: `${job.gpu_ids}`,
      // Memory management for ROCm/HIP on Strix Halo - reduce fragmentation and force device memory
      // Optimized based on rocm-ninodes recommendations for gfx1151
      // Reference: https://github.com/iGavroche/rocm-ninodes
      // expandable_segments helps with memory fragmentation on newer architectures like gfx1151
      // roundup_power2_divisions helps with memory alignment
      // max_split_size_mb:256 is default (reduced from 512 for severe fragmentation)
      // Can be overridden via PYTORCH_MAX_SPLIT_SIZE_MB env var (try 128 or 64 for even more severe fragmentation)
      PYTORCH_CUDA_ALLOC_CONF: process.env.PYTORCH_MAX_SPLIT_SIZE_MB
        ? `expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:${process.env.PYTORCH_MAX_SPLIT_SIZE_MB}`
        : 'expandable_segments:True,roundup_power2_divisions:2,max_split_size_mb:256',
      // ROCm-specific HIP allocator configuration (rocm-ninodes recommendation)
      PYTORCH_HIP_ALLOC_CONF: 'expandable_segments:True',
      // Memory pool type configuration (rocm-ninodes recommendation)
      PYTORCH_CUDA_MEMORY_POOL_TYPE: 'expandable_segments',
      // CRITICAL FIX for unified memory architecture (gfx1151/Strix Halo):
      // Disable PyTorch's caching allocator which doesn't properly account for unified memory
      // The caching allocator reserves memory blocks that it thinks are free but aren't actually
      // available due to unified memory architecture confusion
      // This forces PyTorch to allocate/deallocate memory directly without caching
      // 
      // WARNING: Setting this to '1' causes ROCm driver crashes (0xC0000005) with quantized models
      // If you experience crashes, comment this out (enables caching, may cause fragmentation)
      // PYTORCH_NO_HIP_MEMORY_CACHING: '1',
      // ROCm memory management for Strix Halo (gfx1151) - force dedicated GPU memory usage
      // Disable unified memory to prevent using shared/system memory instead of GPU VRAM
      HSA_XNACK: '0',  // Disable page migration/unified memory (0 = off, 1 = on)
      // Disable SDMA (System DMA) to prevent shared memory transfers
      HSA_ENABLE_SDMA: '0',
      // Force device kernel argument allocation (prevents using system memory for kernel args)
      HIP_FORCE_DEV_KERNARG: '1',
      // Override GFX version for gfx1151 (Strix Halo)
      HSA_OVERRIDE_GFX_VERSION: '11.5.1',
      // Enable memory pool support for better allocation
      HIP_MEM_POOL_SUPPORT: '1',
      // Use 100% of device memory (prevent fallback to shared memory)
      GPU_SINGLE_ALLOC_PERCENT: '100',
      // Set initial device memory size to prioritize GPU memory
      HIP_INITIAL_DM_SIZE: '0',  // 0 = use all available GPU memory
      // Disable peer-to-peer memory access (which can trigger shared memory)
      HIP_ENABLE_PEER_ACCESS: '0',
      IS_AI_TOOLKIT_UI: '1',
    };

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    // Add the --log argument to the command
    const args = [runFilePath, configPath, '--log', logPath];

    try {
      let subprocess;

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: 'ignore', // don't tie stdio to parent
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: 'ignore',
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }

      // Optionally write a pid file for future management (stop/inspect) without keeping streams open
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(subprocess.pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // (No stdout/stderr listeners — logging should go to --log handled by your Python)
      // (No monitoring loop — the whole point is to let it live past this worker)
    } catch (error: any) {
      // Handle any exceptions during process launch
      console.error('Error launching process:', error);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  
  console.log(`Preparing to start job ${jobID} (${job.name}) on GPU(s) ${job.gpu_ids}`);
  
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      info: 'Starting job...',
    },
  });
  
  // start and watch the job asynchronously so the cron can continue
  // Catch any errors in the promise to prevent unhandled rejections
  startAndWatchJob(job).catch(error => {
    console.error(`Error in startAndWatchJob for job ${jobID}:`, error);
    // Update job status to error if the start process fails
    prisma.job.update({
      where: { id: jobID },
      data: {
        status: 'error',
        info: `Error in job startup: ${error instanceof Error ? error.message : String(error)}`,
      },
    }).catch(updateError => {
      console.error(`Error updating job ${jobID} status after startup failure:`, updateError);
    });
  });
}
