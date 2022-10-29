package com.hcifuture.contextactionlibrary.contextaction.action;

import android.content.Context;
import android.hardware.Sensor;
import android.util.Log;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;
import com.hcifuture.contextactionlibrary.contextaction.action.tapfilter.CombinedFilter;
import com.hcifuture.contextactionlibrary.contextaction.action.tapfilter.HorizontalFilter;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.utils.imu.Highpass1C;
import com.hcifuture.contextactionlibrary.utils.imu.Highpass3C;
import com.hcifuture.contextactionlibrary.utils.imu.Lowpass1C;
import com.hcifuture.contextactionlibrary.utils.imu.Lowpass3C;
import com.hcifuture.contextactionlibrary.utils.imu.PeakDetector;
import com.hcifuture.contextactionlibrary.utils.imu.Point3f;
import com.hcifuture.contextactionlibrary.utils.imu.Resample3C;
import com.hcifuture.contextactionlibrary.utils.imu.Sample3C;
import com.hcifuture.contextactionlibrary.utils.imu.Slope3C;
import com.hcifuture.contextactionlibrary.utils.imu.TfClassifier;
import com.hcifuture.contextactionlibrary.utils.imu.Util;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class TapTapAction extends BaseAction {

    private String TAG = "TapTapAction";

    public static String ACTION = "action.taptap.action";
    public static String ACTION_UPLOAD = "action.taptap.action.upload";
    public static String ACTION_RECOGNIZED = "action.taptap.action.recognized";

    private long SAMPLINGINTERVALNS = 2500000L;
    private long WINDOW_NS = 160000000L;

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    protected Highpass3C highpassAcc = new Highpass3C();
    protected Highpass3C highpassGyro = new Highpass3C();
    protected Lowpass3C lowpassAcc = new Lowpass3C();
    protected Lowpass3C lowpassGyro = new Lowpass3C();
    private Resample3C resampleAcc = new Resample3C();
    private Resample3C resampleGyro = new Resample3C();
    protected Slope3C slopeAcc = new Slope3C();
    protected Slope3C slopeGyro = new Slope3C();
    private long syncTime = 0L;
    private List<Float> xsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsAcc = Collections.synchronizedList(new ArrayList<>());
    private List<Float> xsGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> ysGyro = Collections.synchronizedList(new ArrayList<>());
    private List<Float> zsGyro = Collections.synchronizedList(new ArrayList<>());
    private long lastTimestamp = 0L;

    private Highpass1C highpassKey = new Highpass1C();
    private Lowpass1C lowpassKey = new Lowpass1C();
    private PeakDetector peakDetectorPositive = new PeakDetector();
    private boolean wasPeakApproaching = true;
    private int seqLength;
    private Deque<Long> tapTimestamps = new ArrayDeque();
    private TfClassifier tflite;

    // filter related
    private boolean flag1 = true;
    private boolean flag2 = false;
    private boolean existTaptapSignal = false;
    private HorizontalFilter horizontalFilter = new HorizontalFilter();
    private static CombinedFilter combinedFilter = new CombinedFilter();

    private ThreadPoolExecutor threadPoolExecutor;

    public TapTapAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
        init();
        // tflite = new TfClassifier(mContext.getAssets(), "tap7cls_pixel4.tflite");
        tflite = new TfClassifier(new File(ContextActionContainer.getSavePath() + "tap7cls_pixel4.tflite"));
        seqLength = (int)config.getValue("SeqLength");
        threadPoolExecutor = new ThreadPoolExecutor(1, 1, 1000, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(10), Executors.defaultThreadFactory(), new ThreadPoolExecutor.DiscardOldestPolicy());
    }

    private void init() {
        lowpassAcc.setPara(1.0F);
        lowpassGyro.setPara(1.0F);
        highpassAcc.setPara(0.05F);
        highpassGyro.setPara(0.05F);
        lowpassKey.setPara(0.2F);
        highpassKey.setPara(0.2F);
        peakDetectorPositive.setMinNoiseTolerate(0.05f);
        peakDetectorPositive.setWindowSize(64);
    }

    private void reset() {
        gotAcc = false;
        gotGyro = false;
        syncTime = 0L;
        xsAcc.clear();
        ysAcc.clear();
        zsAcc.clear();
        xsGyro.clear();
        ysGyro.clear();
        zsGyro.clear();
        lastTimestamp = 0L;
    }

    @Override
    public synchronized void start() {
        if (isStarted) {
            Log.d(TAG, "Action is already started.");
            return;
        }
        isStarted = true;
        reset();
    }

    @Override
    public synchronized void stop() {
        if (!isStarted) {
            Log.d(TAG, "Action is already stopped");
            return;
        }
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        // just for horizontal / static cases' record && upload
        horizontalFilter.onSensorChanged(data);
        if (horizontalFilter.passWithDelay(data.getTimestamp()) == -1) {
            ActionResult actionResult = new ActionResult(ACTION_UPLOAD);
            actionResult.setReason("Static");
            for (ActionListener listener : actionListener) {
                listener.onAction(actionResult);
            }
        }
//        horizontalFilter.onSensorChanged(event);
//        int tmp1 = horizontalFilter.passWithDelay(event.timestamp);
//        if (tmp1 == -1)
//            existTaptapSignal = false;
//        else if (tmp1 == 1)
//            flag1 = true;
        combinedFilter.onSensorChanged(data);
        int tmp2 = combinedFilter.passWithDelay(data.getTimestamp());
        if (tmp2 == -1)
            existTaptapSignal = false;
        else if (tmp2 == 1)
            flag2 = true;
        if (existTaptapSignal && flag1 && flag2) {
            if (actionListener != null) {
                for (ActionListener listener: actionListener) {
                    listener.onAction(new ActionResult(ACTION));
                }
            }
            existTaptapSignal = false;
//            flag1 = false;
            flag2 = false;
        }

        if (data.getType() != Sensor.TYPE_GYROSCOPE && data.getType() != Sensor.TYPE_ACCELEROMETER)
            return;
        if (data.getType() == Sensor.TYPE_ACCELEROMETER) {
            gotAcc = true;
            if (0L == syncTime)
                resampleAcc.init(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp(), SAMPLINGINTERVALNS);
            if (!gotGyro)
                return;
        } else {
            gotGyro = true;
            if (0L == syncTime)
                resampleGyro.init(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp(), SAMPLINGINTERVALNS);
            if (!gotAcc)
                return;
        }
        if (0L == syncTime) {
            syncTime = data.getTimestamp();
            resampleAcc.setSyncTime(data.getTimestamp());
            resampleGyro.setSyncTime(syncTime);
            slopeAcc.init(resampleAcc.getResults().point);
            slopeGyro.init(resampleGyro.getResults().point);
            lowpassAcc.init(new Point3f(0.0F, 0.0F, 0.0F));
            lowpassGyro.init(new Point3f(0.0F, 0.0F, 0.0F));
            highpassAcc.init(new Point3f(0.0F, 0.0F, 0.0F));
            highpassGyro.init(new Point3f(0.0F, 0.0F, 0.0F));
            lowpassKey.init(0.0F);
            highpassKey.init(0.0F);
        } else {
            if (data.getType() == Sensor.TYPE_ACCELEROMETER)
                while(resampleAcc.update(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp()))
                    processAccAndKeySignal();
            else
                while(resampleGyro.update(data.getValues().get(0), data.getValues().get(1), data.getValues().get(2), data.getTimestamp()))
                    processGyro();
            threadPoolExecutor.execute(() -> {
                recognizeTapML(data.getTimestamp());
            });
        }
    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    private void processAccAndKeySignal() {
        Sample3C sample = resampleAcc.getResults();
        Point3f point1 = slopeAcc.update(sample.point, 2500000.0F / (float)resampleAcc.getInterval());
        Point3f point3 = highpassAcc.update(lowpassAcc.update(point1));
        xsAcc.add(point3.x);
        ysAcc.add(point3.y);
        zsAcc.add(point3.z);
        lastTimestamp = sample.t;
        int size = (int)(WINDOW_NS / resampleAcc.getInterval());

        while(xsAcc.size() > size) {
            xsAcc.remove(0);
            ysAcc.remove(0);
            zsAcc.remove(0);
        }

        peakDetectorPositive.update(highpassKey.update(lowpassKey.update(point1.z)));
    }

    private void processGyro() {
        Point3f point = resampleGyro.getResults().point;
        point = highpassGyro.update(lowpassGyro.update(slopeGyro.update(point, 2500000.0F / (float)resampleGyro.getInterval())));
        xsGyro.add(point.x);
        ysGyro.add(point.y);
        zsGyro.add(point.z);
        int size = (int)(WINDOW_NS / resampleGyro.getInterval());

        while(xsGyro.size() > size) {
            xsGyro.remove(0);
            ysGyro.remove(0);
            zsGyro.remove(0);
        }
    }

    private ArrayList<Float> getInput(int accIdx, int gyroIdx) {
        ArrayList<Float> res = new ArrayList<>();
        addFeatureData(xsAcc, accIdx, 1, res);
        addFeatureData(ysAcc, accIdx, 1, res);
        addFeatureData(zsAcc, accIdx, 1, res);
        addFeatureData(xsGyro, gyroIdx, 10, res);
        addFeatureData(ysGyro, gyroIdx, 10, res);
        addFeatureData(zsGyro, gyroIdx, 10, res);
        return res;
    }

    private void addFeatureData(List<Float> list, int startIdx, int scale, List<Float> res) {
        synchronized (list) {
            for (int i = 0; i < seqLength; i++) {
                if (i + startIdx >= list.size())
                    res.add(0.0F);
                else
                    res.add(scale * list.get(i + startIdx));
            }
        }
    }
    
    private synchronized int checkDoubleTapTiming(long timestamp) {
        Iterator iter = tapTimestamps.iterator();
        while (iter.hasNext()) {
            if (timestamp - (Long)iter.next() > 500000000L) {
                iter.remove();
            }
        }
        int res = 0;
        if (!tapTimestamps.isEmpty()) {
            iter = tapTimestamps.iterator();
            while (true) {
                if (!iter.hasNext()) {
                    res = 1;
                    break;
                }
                if ((Long)tapTimestamps.getLast() - (Long)iter.next() > 100000000L) {
                    tapTimestamps.clear();
                    res = 2;
                    break;
                }
            }
        }
        return res;
    }

    public void recognizeTapML(long timestamp) {
        if (resampleAcc.getInterval() == 0)
            return;
        int result = 0;
        int diff = (int)((resampleAcc.getResults().t - resampleGyro.getResults().t) / resampleAcc.getInterval());
        int peakIdx = peakDetectorPositive.getIdMajorPeak();
        if (peakIdx > 12) {
            wasPeakApproaching = true;
        }
        int accIdx = peakIdx - 6;
        int gyroIdx = accIdx - diff;
        if (accIdx >= 0 && gyroIdx >= 0) {
            if (accIdx + seqLength < zsAcc.size() && gyroIdx + seqLength < zsAcc.size() && wasPeakApproaching && peakIdx <= 12) {
                wasPeakApproaching = false;
                result = Util.getMaxId(tflite.predict(getInput(accIdx, gyroIdx), 7).get(0));
            }
        }
        if (result == 1) {
            tapTimestamps.addLast(timestamp);
            int count = checkDoubleTapTiming(lastTimestamp);
            if (count == 2) {
                if (actionListener != null) {
                    for (ActionListener listener : actionListener) {
                        ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
                        listener.onAction(actionResult);
                    }
                    horizontalFilter.updateCondition();
                    combinedFilter.updateCondition();
                    this.updateCondition();
                }
            }
        }
    }

    private synchronized void updateCondition() {
//        flag1 = false;
        flag2 = false;
        existTaptapSignal = true;
    }

    public synchronized static void onConfirmed() {
        combinedFilter.confirmed();
    }

    @Override
    public synchronized void getAction() {
        /*
        if (!isStarted)
            return;
        int count = checkDoubleTapTiming(lastTimestamp);
        if (count == 2) {
            if (actionListener != null) {
                for (ActionListener listener: actionListener) {
                    ActionResult actionResult = new ActionResult(ACTION_RECOGNIZED);
                    listener.onAction(actionResult);
                }
                horizontalFilter.updateCondition();
                combinedFilter.updateCondition();
                this.updateCondition();
            }
        }
         */
    }
}
