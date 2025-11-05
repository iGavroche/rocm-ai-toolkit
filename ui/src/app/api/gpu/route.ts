import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import os from 'os';

const execAsync = promisify(exec);

// Timeout for command execution (5 seconds)
const COMMAND_TIMEOUT = 5000;

// Execute command with timeout
async function execWithTimeout(command: string, timeout: number = COMMAND_TIMEOUT) {
  return new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Command timed out after ${timeout}ms`));
    }, timeout);

    execAsync(command)
      .then(result => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch(error => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';

    // Detect backend (check ROCm first, then NVIDIA)
    const backend = await detectBackend(isWindows);

    if (backend === 'none') {
      return NextResponse.json({
        backend: 'none',
        gpus: [],
        error: 'No GPU management tool found (neither rocm-smi nor nvidia-smi)',
      });
    }

    // Get GPU stats based on backend
    let gpuStats;
    try {
      if (backend === 'rocm') {
        gpuStats = await getRocmGpuStats(isWindows);
      } else {
        gpuStats = await getNvidiaGpuStats(isWindows);
      }
    } catch (error) {
      console.error(`Error fetching ${backend} GPU stats:`, error);
      return NextResponse.json(
        {
          backend,
          gpus: [],
          error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
        },
        { status: 500 },
      );
    }

    return NextResponse.json({
      backend,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error in GPU API route:', error);
    return NextResponse.json(
      {
        backend: 'none',
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function detectBackend(isWindows: boolean): Promise<'nvidia' | 'rocm' | 'none'> {
  // Check ROCm first
  try {
    if (isWindows) {
      // On Windows, rocm-smi might be in PATH or in ROCm installation directory
      await execWithTimeout('rocm-smi -v', 3000);
      return 'rocm';
    } else {
      // Linux/macOS - check if rocm-smi exists
      const { stdout } = await execWithTimeout('which rocm-smi', 3000);
      if (stdout.trim()) {
        // Verify it works by checking version or product name
        await execWithTimeout('rocm-smi --showproductname', 3000);
        return 'rocm';
      }
    }
  } catch (error) {
    // ROCm not available, continue to check NVIDIA
  }

  // Check NVIDIA
  try {
    if (isWindows) {
      await execWithTimeout('nvidia-smi -L', 3000);
      return 'nvidia';
    } else {
      await execWithTimeout('which nvidia-smi', 3000);
      return 'nvidia';
    }
  } catch (error) {
    // Neither available
    return 'none';
  }
}

async function getNvidiaGpuStats(isWindows: boolean) {
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  // Execute command with timeout
  const { stdout } = await execWithTimeout(command, COMMAND_TIMEOUT);

  // Parse CSV output
  const gpus = stdout
    .trim()
    .split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => {
      const parts = line.split(', ').map(item => item.trim());
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = parts;

      return {
        index: parseInt(index) || 0,
        name: name || 'Unknown GPU',
        driverVersion: driverVersion || undefined,
        temperature: parseInt(temperature) || 0,
        utilization: {
          gpu: parseInt(gpuUtil) || 0,
          memory: parseInt(memoryUtil) || 0,
        },
        memory: {
          total: parseInt(memoryTotal) || 0,
          free: parseInt(memoryFree) || 0,
          used: parseInt(memoryUsed) || 0,
        },
        power: {
          draw: parseFloat(powerDraw) || 0,
          limit: parseFloat(powerLimit) || 0,
        },
        clocks: {
          graphics: parseInt(clockGraphics) || 0,
          memory: parseInt(clockMemory) || 0,
        },
        fan: {
          speed: parseInt(fanSpeed) || 0,
        },
      };
    });

  return gpus;
}

async function getRocmGpuStats(isWindows: boolean) {
  // First, get list of GPUs using --alldevices --showproductname
  let gpuList: number[] = [];
  try {
    const { stdout: listOutput } = await execWithTimeout('rocm-smi --alldevices --showproductname', COMMAND_TIMEOUT);
    // Parse GPU list: "GPU[0]\t: Card Series:\tRadeon 8060S Graphics"
    // Multiple lines per GPU, so we need to extract unique indices
    const gpuMatches = listOutput.match(/GPU\[(\d+)\]/g);
    if (gpuMatches && gpuMatches.length > 0) {
      // Use Set to get unique GPU indices
      const uniqueIndices = new Set<number>();
      gpuMatches.forEach(match => {
        const indexMatch = match.match(/\[(\d+)\]/);
        if (indexMatch) {
          uniqueIndices.add(parseInt(indexMatch[1]));
        }
      });
      gpuList = Array.from(uniqueIndices).sort((a, b) => a - b);
    } else {
      // Fallback: try GPU 0
      console.warn('Could not parse GPU list from rocm-smi, defaulting to GPU 0');
      gpuList = [0];
    }
  } catch (error) {
    console.error('Error getting ROCm GPU list:', error);
    // Try to get at least GPU 0
    gpuList = [0];
  }

  if (gpuList.length === 0) {
    // Fallback: assume at least one GPU (index 0)
    gpuList = [0];
  }

  // Get stats for each GPU
  const gpus = await Promise.all(
    gpuList.map(async (gpuIndex) => {
      try {
        // Get device name
        let name = 'Unknown GPU';
        try {
          const { stdout: nameOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} --showproductname`, COMMAND_TIMEOUT);
          // Parse name from output: "GPU[0]\t: Card Series:\tRadeon 8060S Graphics"
          const nameMatch = nameOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*Card Series:[\s\t]*(.+)/i);
          if (nameMatch) {
            name = nameMatch[2].trim();
          }
        } catch (error) {
          // Continue with default name
        }

        // Get temperature
        let temperature = 0;
        try {
          const { stdout: tempOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} -t`, COMMAND_TIMEOUT);
          // Parse temperature: "GPU[0]\t: Temperature (Sensor edge) (C): 36.0"
          const tempMatch = tempOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*Temperature[^:]*:[\s\t]*(\d+\.?\d*)/i);
          if (tempMatch) {
            temperature = Math.round(parseFloat(tempMatch[2]));
          }
        } catch (error) {
          // Continue with default temperature
        }

        // Get GPU utilization
        let gpuUtil = 0;
        try {
          const { stdout: utilOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} -u`, COMMAND_TIMEOUT);
          // Parse utilization: "GPU[0]\t: GPU use (%): 5"
          const utilMatch = utilOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*GPU use \(%\):[\s\t]*(\d+)/i);
          if (utilMatch) {
            gpuUtil = parseInt(utilMatch[2]);
          }
        } catch (error) {
          // Continue with default utilization
        }

        // Get memory info
        let memoryTotal = 0;
        let memoryUsed = 0;
        let memoryFree = 0;
        try {
          const { stdout: memOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} --showmeminfo VRAM`, COMMAND_TIMEOUT);
          // Parse memory: "GPU[0]\t: VRAM Total Memory (B): 103079215104"
          // Parse memory: "GPU[0]\t: VRAM Total Used Memory (B): 934367232"
          const totalMatch = memOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*VRAM Total Memory \(B\):[\s\t]*(\d+)/i);
          const usedMatch = memOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*VRAM Total Used Memory \(B\):[\s\t]*(\d+)/i);
          
          if (totalMatch) {
            memoryTotal = Math.round(parseInt(totalMatch[2]) / (1024 * 1024)); // Convert bytes to MB
          }
          if (usedMatch) {
            memoryUsed = Math.round(parseInt(usedMatch[2]) / (1024 * 1024)); // Convert bytes to MB
            memoryFree = memoryTotal - memoryUsed;
          }
        } catch (error) {
          // Continue with default memory values
        }

        // Calculate memory utilization percentage
        const memoryUtil = memoryTotal > 0 ? Math.round((memoryUsed / memoryTotal) * 100) : 0;

        // Get clock speeds (optional)
        let clockGraphics = 0;
        let clockMemory = 0;
        try {
          const { stdout: clockOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} -c`, COMMAND_TIMEOUT);
          // Parse clock speeds: "GPU[0]\t: sclk clock level: 1: (981Mhz)"
          // and "GPU[0]\t: mclk clock level: 2: (1000Mhz)"
          const sclkMatch = clockOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*sclk clock level[^:]*:[\s\t]*\((\d+)Mhz\)/i);
          const mclkMatch = clockOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*mclk clock level[^:]*:[\s\t]*\((\d+)Mhz\)/i);
          if (sclkMatch) {
            clockGraphics = parseInt(sclkMatch[2]);
          }
          if (mclkMatch) {
            clockMemory = parseInt(mclkMatch[2]);
          }
        } catch (error) {
          // Clock speeds not critical, continue
        }

        // Get fan speed (optional)
        let fanSpeed = 0;
        try {
          const { stdout: fanOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} -f`, COMMAND_TIMEOUT);
          // Parse fan speed - may show "Not supported"
          const fanMatch = fanOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*(\d+)/);
          if (fanMatch && !fanOutput.includes('Not supported')) {
            fanSpeed = parseInt(fanMatch[2]);
          }
        } catch (error) {
          // Fan speed not critical, continue
        }

        // Get power info (optional)
        let powerDraw = 0;
        let powerLimit = 0;
        try {
          const { stdout: powerOutput } = await execWithTimeout(`rocm-smi -d ${gpuIndex} -P`, COMMAND_TIMEOUT);
          // Parse power: "GPU[0]\t: Current Socket Graphics Package Power (W): 26.096"
          const drawMatch = powerOutput.match(/GPU\[(\d+)\][\s\t]*:[\s\t]*Current[^:]*Power \(W\):[\s\t]*(\d+\.?\d*)/i);
          if (drawMatch) {
            powerDraw = parseFloat(drawMatch[2]);
          }
          // Power limit might be in a different command or not available
        } catch (error) {
          // Power info not critical, continue
        }

        return {
          index: gpuIndex,
          name,
          driverVersion: undefined, // ROCm doesn't provide driver version in same format
          temperature,
          utilization: {
            gpu: gpuUtil,
            memory: memoryUtil,
          },
          memory: {
            total: memoryTotal,
            free: memoryFree,
            used: memoryUsed,
          },
          power: powerDraw > 0 || powerLimit > 0 ? {
            draw: powerDraw,
            limit: powerLimit,
          } : undefined,
          clocks: clockGraphics > 0 || clockMemory > 0 ? {
            graphics: clockGraphics,
            memory: clockMemory,
          } : undefined,
          fan: fanSpeed > 0 ? {
            speed: fanSpeed,
          } : undefined,
        };
      } catch (error) {
        console.error(`Error getting stats for ROCm GPU ${gpuIndex}:`, error);
        // Return minimal GPU info
        return {
          index: gpuIndex,
          name: 'Unknown GPU',
          temperature: 0,
          utilization: {
            gpu: 0,
            memory: 0,
          },
          memory: {
            total: 0,
            free: 0,
            used: 0,
          },
        };
      }
    })
  );

  return gpus;
}
