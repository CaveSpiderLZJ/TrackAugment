package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.Collector;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public abstract class AsynchronousCollector extends Collector {

    public AsynchronousCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    public abstract CompletableFuture<CollectorResult> getData(TriggerConfig config);

    // public abstract CompletableFuture<String> getDataString(TriggerConfig config);
}
