import prisma from '../prisma';

import { Job, Queue } from '@prisma/client';
import startJob from './startJob';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '../paths';

export default async function processQueue() {
  const queues: Queue[] = await prisma.queue.findMany({
    orderBy: {
      id: 'asc',
    },
  });

  for (const queue of queues) {
    // Normalize queue GPU ID - remove "gpu_" prefix if present
    let normalizedQueueGpuId = queue.gpu_ids;
    if (normalizedQueueGpuId.startsWith('gpu_')) {
      normalizedQueueGpuId = normalizedQueueGpuId.replace('gpu_', '');
    }
    
    if (!queue.is_running) {
      // stop any running jobs first - try normalized ID
      let runningJobs: Job[] = await prisma.job.findMany({
        where: {
          status: 'running',
          gpu_ids: normalizedQueueGpuId,
        },
      });
      
      // Also check jobs with comma-separated GPU IDs
      if (runningJobs.length === 0) {
        const allRunningJobs = await prisma.job.findMany({
          where: {
            status: 'running',
          },
        });
        runningJobs = allRunningJobs.filter(job => {
          if (!job.gpu_ids) return false;
          const jobGpuIds = job.gpu_ids.split(',').map(id => id.trim());
          return jobGpuIds.includes(normalizedQueueGpuId);
        });
      }

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await prisma.job.update({
          where: { id: job.id },
          data: {
            return_to_queue: true,
            info: 'Stopping job...',
          },
        });
      }
    }
    if (queue.is_running) {
      // first see if one is already running, status of running or stopping - try normalized ID
      let runningJob: Job | null = await prisma.job.findFirst({
        where: {
          status: { in: ['running', 'stopping'] },
          gpu_ids: normalizedQueueGpuId,
        },
      });
      
      // Also check jobs with comma-separated GPU IDs
      if (!runningJob) {
        const allRunningJobs = await prisma.job.findMany({
          where: {
            status: { in: ['running', 'stopping'] },
          },
        });
        runningJob = allRunningJobs.find(job => {
          if (!job.gpu_ids) return false;
          const jobGpuIds = job.gpu_ids.split(',').map(id => id.trim());
          return jobGpuIds.includes(normalizedQueueGpuId);
        }) || null;
      }

      if (runningJob) {
        // Check if job is actually still running (process might have crashed)
        // We'll check the log file's last modified time as a simple check
        // If it's been more than 5 minutes without an update, assume it crashed
        const trainingRoot = await getTrainingFolder();
        const trainingFolder = path.join(trainingRoot, runningJob.name);
        const logPath = path.join(trainingFolder, 'log.txt');
        
        try {
          if (fs.existsSync(logPath)) {
            const stats = fs.statSync(logPath);
            const now = Date.now();
            const lastModified = stats.mtime.getTime();
            const minutesSinceUpdate = (now - lastModified) / (1000 * 60);
            
            // If log hasn't been updated in 5 minutes and job has been running for more than 5 minutes
            if (minutesSinceUpdate > 5 && runningJob.status === 'running') {
              console.log(`Job ${runningJob.id} appears to have stopped (log not updated in ${minutesSinceUpdate.toFixed(1)} minutes)`);
              // Mark as error so queue can continue
              await prisma.job.update({
                where: { id: runningJob.id },
                data: {
                  status: 'error',
                  info: `Job appears to have stopped (log not updated in ${minutesSinceUpdate.toFixed(1)} minutes)`,
                },
              });
              // Continue to process next job
            } else {
              // already running normally, nothing to do
              continue; // skip to next queue
            }
          } else {
            // Log file doesn't exist yet, job might be starting
            // Check if PID file exists (process was spawned)
            const pidPath = path.join(trainingFolder, 'pid.txt');
            const hasPidFile = fs.existsSync(pidPath);
            
            // Use updated_at instead of created_at to check when job actually started running
            const jobAge = Date.now() - runningJob.updated_at.getTime();
            const minutesSinceStart = jobAge / (1000 * 60);
            
            // Give jobs more time if they have a PID file (process was spawned)
            // Increased timeout for dataset preparation and model loading which can take longer
            // Especially on ROCm/gfx1151 where memory management takes additional time
            const timeoutMinutes = hasPidFile ? 15 : 10;  // Increased from 5/2 to 15/10 minutes
            
            if (minutesSinceStart > timeoutMinutes && runningJob.status === 'running') {
              console.log(`Job ${runningJob.id} has no log file after ${minutesSinceStart.toFixed(1)} minutes (PID file: ${hasPidFile ? 'exists' : 'missing'}), marking as error`);
              await prisma.job.update({
                where: { id: runningJob.id },
                data: {
                  status: 'error',
                  info: `Job appears to have failed to start (no log file created after ${minutesSinceStart.toFixed(1)} minutes)`,
                },
              });
              // Continue to process next job
            } else {
              // Job is still starting, wait a bit
              continue;
            }
          }
        } catch (error) {
          console.error(`Error checking job ${runningJob.id} status:`, error);
          // If we can't check, assume it's running and skip
          continue;
        }
      } else {
        // find the next job in the queue - use normalized ID (already normalized above)
        let nextJob: Job | null = await prisma.job.findFirst({
          where: {
            status: 'queued',
            gpu_ids: normalizedQueueGpuId,
          },
          orderBy: {
            queue_position: 'asc',
          },
        });
        
        // If no job found with normalized ID, try the original queue.gpu_ids
        if (!nextJob) {
          nextJob = await prisma.job.findFirst({
            where: {
              status: 'queued',
              gpu_ids: queue.gpu_ids,
            },
            orderBy: {
              queue_position: 'asc',
            },
          });
        }
        
        // Also try to find jobs with gpu_ids that start with the normalized ID (for comma-separated lists)
        if (!nextJob) {
          const allQueuedJobs = await prisma.job.findMany({
            where: {
              status: 'queued',
            },
            orderBy: {
              queue_position: 'asc',
            },
          });
          
          nextJob = allQueuedJobs.find(job => {
            if (!job.gpu_ids) return false;
            const jobGpuIds = job.gpu_ids.split(',').map(id => id.trim());
            return jobGpuIds.includes(normalizedQueueGpuId);
          }) || null;
        }
        if (nextJob) {
          console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
          try {
            await startJob(nextJob.id);
            console.log(`Job ${nextJob.id} start command completed`);
          } catch (error) {
            console.error(`Error starting job ${nextJob.id}:`, error);
            // Mark job as error and continue processing queue
            await prisma.job.update({
              where: { id: nextJob.id },
              data: {
                status: 'error',
                info: `Error starting job: ${error instanceof Error ? error.message : String(error)}`,
              },
            });
          }
        } else {
          // no more jobs, stop the queue - but first check what jobs exist for debugging
          const allQueuedJobs = await prisma.job.findMany({
            where: { status: 'queued' },
          });
          const allStoppedJobs = await prisma.job.findMany({
            where: { status: 'stopped' },
          });
          console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids} (normalized: ${normalizedQueueGpuId})`);
          console.log(`  Total queued jobs: ${allQueuedJobs.length}`);
          console.log(`  Total stopped jobs: ${allStoppedJobs.length}`);
          if (allQueuedJobs.length > 0) {
            console.log(`  Queued jobs GPU IDs:`, allQueuedJobs.map(j => j.gpu_ids).join(', '));
          }
          if (allStoppedJobs.length > 0) {
            console.log(`  Stopped jobs GPU IDs:`, allStoppedJobs.map(j => j.gpu_ids).join(', '));
            console.log(`  Note: Stopped jobs won't be processed until they're queued. Start them manually or they'll be auto-queued when the queue starts.`);
          }
          await prisma.queue.update({
            where: { id: queue.id },
            data: { is_running: false },
          });
        }
      }
    }
  }
}
