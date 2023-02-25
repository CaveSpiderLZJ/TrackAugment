package com.hcifuture.datacollection.data;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Get data from four sensors: gyroscope, linear acceleration sensor,
 * accelerometer and magnetic field sensor.
 * Save the sensor data to files and the backend.
 */
public class MotionSensorController {
    // sensor
    private SensorManager mSensorManager;
    private int mSamplingMode = SensorManager.SENSOR_DELAY_FASTEST;  // fastest

    private Sensor mAccSensor;          // 3D
    private Sensor mAccUnSensor;        // 6D
    private Sensor mGyroSensor;         // 3D
    private Sensor mGyroUnSensor;       // 6D
    private Sensor mMagSensor;          // 3D
    private Sensor mMagUnSensor;        // 6D
    private Sensor mLinearAccSensor;    // 3D
    private Sensor mGravitySensor;      // 3D
    private Sensor mRotationSensor;     // 4D

    private Context mContext;

    private List<SensorData> mAccData = new ArrayList<>();
    private List<SensorData> mAccUnData = new ArrayList<>();
    private List<SensorData> mGyroData = new ArrayList<>();
    private List<SensorData> mGyroUnData = new ArrayList<>();
    private List<SensorData> mMagData = new ArrayList<>();
    private List<SensorData> mMagUnData = new ArrayList<>();
    private List<SensorData> mLinearAccData = new ArrayList<>();
    private List<SensorData> mGravityData = new ArrayList<>();
    private List<SensorData> mRotationData = new ArrayList<>();

    private long mLastTimestamp;

    private File mSensorFile;
    private SensorEventListener mListener;

    /**
     * Constructor.
     * Initialize the four sensors: gyro, linear, acc and mag and check if they are
     * successfully gotten.
     * @param context the current application context.
     */
    public MotionSensorController(Context context) {
        this.mContext = context;

        mSensorManager = (SensorManager) mContext.getSystemService(Context.SENSOR_SERVICE);

        mAccSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mAccUnSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED);
        mGyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mGyroUnSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        mMagSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        mMagUnSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED);
        mLinearAccSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGravitySensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        mRotationSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

        mListener = new SensorEventListener() {
            /**
             * Save data in one sampling.
             * Q: Why save all four sensor data to only one sensor data array ???
             * @param event the SensorEvent passed to the listener
             */
            @Override
            public void onSensorChanged(SensorEvent event) {
                int type = event.sensor.getType();
                float[] values = event.values;
                long timestamp = event.timestamp;
                if (type == Sensor.TYPE_ACCELEROMETER) {
                    mAccData.add(new SensorData(3, values, timestamp));
                } else if (type == Sensor.TYPE_ACCELEROMETER_UNCALIBRATED) {
                    mAccUnData.add(new SensorData(6, values, timestamp));
                } else if (type == Sensor.TYPE_GYROSCOPE) {
                    mGyroData.add(new SensorData(3, values, timestamp));
                } else if (type == Sensor.TYPE_GYROSCOPE_UNCALIBRATED) {
                    mGyroUnData.add(new SensorData(6, values, timestamp));
                } else if (type == Sensor.TYPE_MAGNETIC_FIELD) {
                    mMagData.add(new SensorData(3, values, timestamp));
                } else if (type == Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED) {
                    mMagUnData.add(new SensorData(6, values, timestamp));
                } else if (type == Sensor.TYPE_LINEAR_ACCELERATION) {
                    mLinearAccData.add(new SensorData(3, values, timestamp));
                } else if (type == Sensor.TYPE_GRAVITY) {
                    mGravityData.add(new SensorData(3, values, timestamp));
                } else if (type == Sensor.TYPE_ROTATION_VECTOR) {
                    mRotationData.add(new SensorData(4, values, timestamp));
                }
                mLastTimestamp = event.timestamp;
            }
            /**
             * Not implemented yet.
             * @param sensor
             * @param i
             */
            @Override
            public void onAccuracyChanged(Sensor sensor, int i) {}
        };

        if (!isSensorSupport()) {
            Log.w("SensorController", "Sensor missing!");
        }
    }

    /**
     * Called when the user want to start recording IMU data.
     * Init the sensorFile and sensorBinFile and call resume() to register the listener
     * @param file The motion sensor file.
     */
    public void start(File file) {
        mSensorFile = file;
        clearData();

        if (mSensorManager != null) {
            mSensorManager.registerListener(mListener, mAccSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mAccUnSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mGyroSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mGyroUnSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mMagSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mMagUnSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mLinearAccSensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mGravitySensor, mSamplingMode);
            mSensorManager.registerListener(mListener, mRotationSensor, mSamplingMode);
        }
    }

    /**
     * Called when the user want to cancel an ongoing subtask.
     * Cancel recording the data by unregister the listener.
     */
    public void cancel() {
        if (mSensorManager != null) mSensorManager.unregisterListener(mListener);
        clearData();
    }

    /**
     * Called when a whole subtask is recorded.
     * Unregister the listener and write all data to files.
     */
    public void stop() {
        if (mSensorManager != null) mSensorManager.unregisterListener(mListener);
        ArrayList<List<SensorData>> sensorData = new ArrayList<>(Arrays.asList(
                mAccData, mAccUnData, mGyroData, mGyroUnData, mMagData, mMagUnData,
                mLinearAccData, mGravityData, mRotationData));
        FileUtils.writeIMUDataToFile2(sensorData, mSensorFile);
        clearData();
    }

    /**
     * Check if all sensors were successfully gotten.
     * @return boolean
     */
    public boolean isSensorSupport() {
        return (mAccSensor != null && mAccUnSensor != null && mGyroSensor != null &&
                mGyroUnSensor != null && mMagSensor != null && mMagUnSensor != null &&
                mLinearAccSensor != null && mGravitySensor != null && mRotationSensor != null);
    }

    public long getLastTimestamp() {
        return mLastTimestamp;
    }

    public void clearData() {
        mAccData.clear(); mGyroData.clear(); mMagData.clear();
        mAccUnData.clear(); mGyroUnData.clear(); mMagUnData.clear();
        mLinearAccData.clear(); mGravityData.clear(); mRotationData.clear();
    }

    /**
     * Upload data files to the backend.
     * @param taskListId
     * @param taskId
     * @param subtaskId
     * @param recordId
     * @param timestamp
     */
    public void upload(String taskListId, String taskId, String subtaskId,
                       String recordId, long timestamp) {
        if (mSensorFile != null) {
            NetworkUtils.uploadRecordFile(mContext, mSensorFile,
                    RootListBean.FILE_TYPE.MOTION.ordinal(), taskListId, taskId,
                    subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) { }
            });
        }
    }
}
