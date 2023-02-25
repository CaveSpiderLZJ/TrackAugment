package com.hcifuture.datacollection.data;

import android.content.Context;
import android.os.CountDownTimer;
import android.os.Handler;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.RandomUtils;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * A very important class for managing sensors, used in MainActivity.
 */
public class Recorder {

    private Context mContext;

    private MotionSensorController mMotionSensorController;
    private TimestampController mTimestampController;

    // each of the following file will be passed to the start() function
    // of the corresponding sensor controller and used in it
    private File mMotionSensorFile;
    private File mTimestampFile;

    private RootListBean mRootList;
    private RootListBean.TaskList mTaskList;
    private RootListBean.TaskList.Task mTask;
    private RootListBean.TaskList.Task.Subtask mSubtask;
    private CountDownTimer mTimer;
    private RecorderListener mListener;

    private String mRecordId;
    private int mTickCount = 0;
    private final SimpleDateFormat mDateFormat = new SimpleDateFormat("yyMMddHHmmss");
    private String mUserName; // user name set when starting recording

    public Recorder(Context context, RecorderListener listener) {
        this.mContext = context;
        this.mListener = listener;
        mMotionSensorController = new MotionSensorController(mContext);
        mTimestampController = new TimestampController(mContext);
        mUserName = "DefaultUser";
        FileUtils.makeDir(BuildConfig.SAVE_PATH);
    }

    public void start(String userName, int taskListId, int taskId, int subtaskId, RootListBean rootList) {
        mRootList = rootList;
        mTaskList = mRootList.getTaskLists().get(taskListId);
        mTask = mTaskList.getTasks().get(taskId);
        mSubtask = mTask.getSubtasks().get(subtaskId);
        mTickCount = 0;
        mRecordId = RandomUtils.generateRandomRecordId();
        mUserName = userName;

        createFile(taskId, subtaskId);

        long duration = mSubtask.getDuration();
        int times = mSubtask.getTimes();
        long actionTime = times * mSubtask.getDuration();

        if (mTimer != null) mTimer.cancel();
        mTimer = new CountDownTimer(actionTime, duration) {
            @Override
            public void onTick(long l) {
                // skip first tick
                if (l < duration / 10) return;
                mTimestampController.add(mMotionSensorController.getLastTimestamp());
                mTickCount += 1;
                mListener.onTick(mTickCount, times);
            }

            @Override
            public void onFinish() {
                mListener.onFinish();
                stop();
            }
        };

        // canceled Handler().postDelayed(() -> {...});
        mMotionSensorController.start(mMotionSensorFile);
        mTimestampController.start(mTimestampFile);
        mTimer.start();
    }

    /**
     * Called when the user want to cancel an ongoing subtask.
     * Cancel all sensors as if this subtask has never been started.
     */
    public void cancel() {
        if (mTimer != null) mTimer.cancel();
        mMotionSensorController.cancel();
        mTimestampController.cancel();
    }

    private void stop() {
        if (mTimer != null) mTimer.cancel();
        mMotionSensorController.stop();
        mTimestampController.stop();

        NetworkUtils.addRecord(mContext, mTaskList.getId(), mTask.getId(), mSubtask.getId(),
                mUserName, mRecordId, System.currentTimeMillis(), new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {}
        });

        new Handler().postDelayed(() -> {
            long timestamp = System.currentTimeMillis();
            mMotionSensorController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
            mTimestampController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
        }, 2000);
    }

    public void createFile(int taskId, int subtaskId) {
        String suffix = mUserName + "_" + taskId + "_" + subtaskId + "_" + mDateFormat.format(new Date());
        mTimestampFile = new File(BuildConfig.SAVE_PATH, "Timestamp_" + suffix + ".json");
        mMotionSensorFile = new File(BuildConfig.SAVE_PATH, "Motion_" + suffix + ".bin");
    }

    public interface RecorderListener {
        void onTick(int tickCount, int times);
        void onFinish();
    }
}
