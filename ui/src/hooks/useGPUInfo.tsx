'use client';

import { GPUApiResponse, GpuInfo } from '@/types';
import { useEffect, useState, useRef, useCallback } from 'react';
import { apiClient } from '@/utils/api';

export default function useGPUInfo(gpuIds: null | number[] = null, reloadInterval: null | number = null) {
  const [gpuList, setGpuList] = useState<GpuInfo[]>([]);
  const [isGPUInfoLoaded, setIsLoaded] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  
  // Refs for error recovery and rate limiting
  const errorCountRef = useRef(0);
  const lastFetchTimeRef = useRef(0);
  const isFetchingRef = useRef(false);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Rate limiting: minimum 500ms between requests
  const MIN_FETCH_INTERVAL = 500;

  const fetchGpuInfo = useCallback(async () => {
    // Prevent concurrent fetches
    if (isFetchingRef.current) {
      return;
    }

    // Rate limiting
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchTimeRef.current;
    if (timeSinceLastFetch < MIN_FETCH_INTERVAL) {
      return;
    }

    isFetchingRef.current = true;
    lastFetchTimeRef.current = now;
    setStatus('loading');

    try {
      const data: GPUApiResponse = await apiClient.get('/api/gpu').then(res => res.data);
      
      // Reset error count on success
      errorCountRef.current = 0;
      
      let gpus = data.gpus.sort((a, b) => a.index - b.index);
      if (gpuIds) {
        gpus = gpus.filter(gpu => gpuIds.includes(gpu.index));
      }
      setGpuList(gpus);
      setStatus('success');
    } catch (err) {
      errorCountRef.current += 1;
      const errorMessage = err instanceof Error ? err.message : String(err);
      console.error(`Failed to fetch GPU data (attempt ${errorCountRef.current}): ${errorMessage}`);
      
      // Exponential backoff: wait 2^errorCount seconds before retry
      const backoffDelay = Math.min(1000 * Math.pow(2, errorCountRef.current), 30000); // Max 30 seconds
      
      setStatus('error');
      
      // Only retry if we haven't exceeded max retries
      if (errorCountRef.current < 5) {
        if (retryTimeoutRef.current) {
          clearTimeout(retryTimeoutRef.current);
        }
        retryTimeoutRef.current = setTimeout(() => {
          fetchGpuInfo();
        }, backoffDelay);
      }
    } finally {
      setIsLoaded(true);
      isFetchingRef.current = false;
    }
  }, [gpuIds]);

  useEffect(() => {
    // Fetch immediately on component mount
    fetchGpuInfo();

    // Set up interval if specified
    let interval: NodeJS.Timeout | null = null;
    if (reloadInterval && reloadInterval > 0) {
      interval = setInterval(() => {
        fetchGpuInfo();
      }, Math.max(reloadInterval, MIN_FETCH_INTERVAL));
    }

    // Cleanup on unmount
    return () => {
      if (interval) {
        clearInterval(interval);
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [gpuIds, reloadInterval, fetchGpuInfo]);

  return { gpuList, setGpuList, isGPUInfoLoaded, status, refreshGpuInfo: fetchGpuInfo };
}
