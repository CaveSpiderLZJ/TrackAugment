package com.hcifuture.contextactionlibrary.contextaction.collect;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.action.PocketAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TapTapAction;
import com.hcifuture.contextactionlibrary.contextaction.action.TopTapAction;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.trigger.ClickTrigger;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.sensor.uploader.Uploader;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;
import com.hcifuture.shared.communicate.result.ContextResult;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class TapTapCollector extends BaseCollector {
    private String markTimestamp;
    private CompletableFuture<List<CollectorResult>> lastFuture;

    public TapTapCollector(Context context, ScheduledExecutorService scheduledExecutorService,
                           List<ScheduledFuture<?>> futureList, RequestListener requestListener,
                           ClickTrigger clickTrigger, Uploader uploader) {
        super(context, scheduledExecutorService, futureList, requestListener, clickTrigger, uploader);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onAction(ActionResult action) {
        if (action.getAction().equals(TapTapAction.ACTION) || action.getAction().equals(TopTapAction.ACTION) || action.getAction().equals(PocketAction.ACTION)) {
            lastFuture = clickTrigger.trigger(Collections.singletonList(CollectorManager.CollectorType.IMU), new TriggerConfig());
        } else if (action.getAction().equals(TapTapAction.ACTION_UPLOAD) || action.getAction().equals(TopTapAction.ACTION_UPLOAD) || action.getAction().equals(PocketAction.ACTION_UPLOAD)) {
            if (lastFuture != null) {
                String name = action.getAction();
                String commit = action.getAction() + ":" + action.getReason() + ":" + markTimestamp;
                if (lastFuture.isDone()) {
                    try {
                        upload(lastFuture.get().get(0), name, commit);
                    } catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    lastFuture.whenComplete((v, e) -> upload(v.get(0), name, commit));
                }
            }
        } else if (action.getAction().equals(TopTapAction.ACTION_RECOGNIZED) || action.getAction().equals(PocketAction.ACTION_RECOGNIZED)) {
//            markTimestamp = action.getTimestamp();
            long[] tmp = action.getExtras().getLongArray("timestamp");
            markTimestamp = tmp[0] + ":" + tmp[1];
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void onContext(ContextResult context) {
        if (!context.getContext().equals("UserAction")) {
            return;
        }
        if (clickTrigger != null) {
            triggerAndUpload(CollectorManager.CollectorType.IMU,
                    new TriggerConfig().setImuHead(800).setImuTail(200),
                    context.getContext(),
                    context.getContext() + ":" + context.getTimestamp()
            );
        }
    }
}
