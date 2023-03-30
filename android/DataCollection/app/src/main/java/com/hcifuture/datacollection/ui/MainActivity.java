package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.provider.Settings;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.inference.Inferencer;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.data.Recorder;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.EasyPermissions;
import pub.devrel.easypermissions.PermissionRequest;

public class MainActivity extends AppCompatActivity {

    private Context mContext;
    private AppCompatActivity mActivity;
    private Vibrator mVibrator;

    // ui
    private EditText mUserText;
    private Spinner mTaskListSpinner;
    private ArrayAdapter<String> mTaskListAdapter;
    private Spinner mTaskSpinner;
    private ArrayAdapter<String> mTaskAdapter;
    private Spinner mSubtaskSpinner;
    private ArrayAdapter<String> mSubtaskAdapter;
    private TextView mTaskDescription;
    private TextView mTaskCounter;
    private Button mBtnStart;
    private Button mBtnCancel;
    private Button mBtnConfig;

    // task
    private RootListBean mRootList;  // queried from the backend
    private String[] mTaskListNames;
    private String[] mTaskNames;
    private String[] mSubtaskNames;
    private int mTaskListIdx = 0;
    private int mTaskIdx = 0;
    private int mSubtaskIdx = 0;
    private int mCurrentTic = 0;    // tic showed in task counter
    private int mTotalTics = 0;      // mCurrentTic / mTotalTic
    private Recorder mRecorder;

    // permission
    private static final int RC_PERMISSIONS = 0;
    private String[] mPermissions = new String[]{
            Manifest.permission.INTERNET,
            Manifest.permission.VIBRATE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACTIVITY_RECOGNITION
    };

    private Inferencer mInferencer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // ask for permissions
        requestPermissions();

        mContext = this;
        mActivity = this;

        // vibrate to indicate data collection progress
        mVibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        mVibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);

        mRecorder = new Recorder(this, new Recorder.RecorderListener() {
            @Override
            public void onTick(int tickCount, int times) {
                mTaskCounter.setText(tickCount + " / " + times);
                mVibrator.vibrate(VibrationEffect.createOneShot(100, 64));
            }

            @Override
            public void onFinish() {
                mVibrator.vibrate(VibrationEffect.createOneShot(400, 64));
                enableButtons(false);
                mCurrentTic = 0;
                updateTaskCounter();
            }
        });

        // find views
        mUserText = findViewById(R.id.user_text);


        mInferencer = Inferencer.getInstance();
        mInferencer.start(this);

        mUserText = findViewById(R.id.user_text);
    }

    private void loadRootListViaNetwork() {
        NetworkUtils.getRootList(this, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mRootList = new Gson().fromJson(response.body(), RootListBean.class);
                initView();
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        mUserText.clearFocus();
        loadRootListViaNetwork();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // Forward results to EasyPermissions
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    /**
     * Pop up dialog windows to ask users for system permissions.
     */
    @AfterPermissionGranted(RC_PERMISSIONS)
    private void requestPermissions() {
        if (!EasyPermissions.hasPermissions(this, mPermissions)) {
            // no permissions, request dynamically
            EasyPermissions.requestPermissions(new PermissionRequest.Builder(
                    this, RC_PERMISSIONS, mPermissions)
                    .setRationale(R.string.rationale)
                    .setPositiveButtonText(R.string.rationale_ask_ok)
                    .setNegativeButtonText(R.string.rationale_ask_cancel)
                    .setTheme(R.style.Theme_AppCompat).build());
        }
    }

    /**
     * Init the status of all UI components in main activity.
     * Called in loadRootListViaNetwork().
     */
    private void initView() {

        // init views
        mTaskDescription = findViewById(R.id.task_description);
        mTaskCounter = findViewById(R.id.task_counter);
        mTaskListSpinner = findViewById(R.id.task_list_spinner);
        mTaskSpinner = findViewById(R.id.task_spinner);
        mSubtaskSpinner = findViewById(R.id.subtask_spinner);

        // task list spinner
        mTaskListNames = mRootList.getTaskListNames();
        mTaskListAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, mTaskListNames);
        mTaskListAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mTaskListSpinner.setAdapter(mTaskListAdapter);
        mTaskListSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mTaskListIdx = position;
                mTaskNames = mRootList.getTaskLists().get(mTaskListIdx).getTaskNames();
                mTaskAdapter = new ArrayAdapter<>(mContext, android.R.layout.simple_spinner_item, mTaskNames);
                mTaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                mTaskSpinner.setAdapter(mTaskAdapter);
                // TODO: check if subtask will change accordingly
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // task spinner
        if (mTaskListNames.length == 0) mTaskNames = new String[0];
        else mTaskNames = mRootList.getTaskLists().get(mTaskListIdx).getTaskNames();
        mTaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, mTaskListNames);
        mTaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mTaskSpinner.setAdapter(mTaskAdapter);
        mTaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mTaskIdx = position;
                mSubtaskNames = mRootList.getTaskLists().get(mTaskListIdx)
                        .getTasks().get(mTaskIdx).getSubtaskNames();
                mSubtaskAdapter = new ArrayAdapter<>(mContext, android.R.layout.simple_spinner_item, mSubtaskNames);
                mSubtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                mSubtaskSpinner.setAdapter(mSubtaskAdapter);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // subtask spinner
        if (mTaskNames.length == 0) mSubtaskNames = new String[0];
        else mSubtaskNames = mRootList.getTaskLists().get(mTaskListIdx)
                .getTasks().get(mTaskIdx).getSubtaskNames();
        mSubtaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, mSubtaskNames);
        mSubtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mSubtaskSpinner.setAdapter(mSubtaskAdapter);
        mSubtaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mRecorder.cancel();
                enableButtons(false);
                mSubtaskIdx = position;
                // set task description with the subtask name
                String taskName = mRootList.getTaskLists()
                        .get(mTaskListIdx).getTasks().get(mTaskIdx).getName();
                String subtaskName = mRootList.getTaskLists().get(mTaskListIdx).getTasks().
                        get(mTaskIdx).getSubtasks().get(mSubtaskIdx).getName();
                mTaskDescription.setText(taskName + "." + subtaskName);
                // init the task counter when subtask selected
                RootListBean.TaskList.Task.Subtask currentSubtask = mRootList
                        .getTaskLists().get(mTaskListIdx)
                        .getTasks().get(mTaskIdx)
                        .getSubtasks().get(mSubtaskIdx);
                mCurrentTic = 0;
                mTotalTics = currentSubtask.getTimes();
                updateTaskCounter();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // btn start and cancel
        mBtnStart = findViewById(R.id.btn_start);
        mBtnCancel = findViewById(R.id.btn_cancel);
        mBtnConfig = findViewById(R.id.btn_config);

        // click the start button to start recorder
        mBtnStart.setOnClickListener(view -> {
            if (mTaskListIdx < mRootList.getTaskLists().size() &&
                    mTaskIdx < mRootList.getTaskLists().get(mTaskListIdx).getTasks().size() &&
                    mSubtaskIdx < mRootList.getTaskLists().get(mTaskListIdx).getTasks()
                            .get(mTaskIdx).getSubtasks().size()) {
                enableButtons(true);
                mRecorder.start(mUserText.getText().toString(), mTaskListIdx,
                        mTaskIdx, mSubtaskIdx, mRootList);
            }
        });

        // click the stop button to end recording
        mBtnCancel.setOnClickListener(view -> {
            enableButtons(false);
            mRecorder.cancel();
            mCurrentTic = 0;
            updateTaskCounter();
        });

        // goto config task activity
        mBtnConfig.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, ConfigTaskListActivity.class);
            startActivity(intent);
        });

        // set the default status of the start and end buttons
        enableButtons(false);
    }

    /**
     * Set the availability of the start and stop buttons.
     * Ensures the status of these two buttons are opposite.
     * @param isRecording Whether the current task is ongoing.
     */
    private void enableButtons(boolean isRecording) {
        mBtnStart.setEnabled(!isRecording);
        mBtnCancel.setEnabled(isRecording);
    }

    /**
     * Cancel the vibrator.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mVibrator != null) {
            mVibrator.cancel();
        }
    }

    private void updateTaskCounter() {
        if (mTaskCounter == null) return;
        mTaskCounter.setText(mCurrentTic + " / " + mTotalTics);
    }
}